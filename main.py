import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db
from database import create_document

# Import schema metadata for external viewers
import schemas

app = FastAPI(title="Sola API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Utilities --------------------

def iso_today() -> str:
    # Use UTC as default day boundary. Frontend can pass explicit date if needed.
    return datetime.now(timezone.utc).date().isoformat()


def to_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    out = {}
    for k, v in doc.items():
        if k == "_id":
            out["id"] = str(v)
        elif isinstance(v, (datetime,)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def get_xpstate() -> Dict[str, Any]:
    state = db["xpstate"].find_one({})
    if not state:
        # initialize
        state = {
            "total_xp": 0,
            "level": 1,
            "xp_in_level": 0,
            "xp_for_next": 100,
            "streak": 0,
            "missions_finished": 0,
            "mood_logs": 0,
            "habit_completions": 0,
        }
        db["xpstate"].insert_one(state)
        state = db["xpstate"].find_one({})
    return state


def add_xp(amount: int) -> Dict[str, Any]:
    state = get_xpstate()
    state["total_xp"] += amount
    state["xp_in_level"] += amount
    # Fixed 100 XP per level
    while state["xp_in_level"] >= state["xp_for_next"]:
        state["xp_in_level"] -= state["xp_for_next"]
        state["level"] += 1
        # xp_for_next stays 100 (fixed)
        # Achievement for reaching specific levels will be checked below
    db["xpstate"].update_one({"_id": state["_id"]}, {"$set": state})
    check_achievements()
    return state


def set_streak(new_streak: int):
    state = get_xpstate()
    state["streak"] = new_streak
    db["xpstate"].update_one({"_id": state["_id"]}, {"$set": {"streak": new_streak}})
    check_achievements()


def increment_counter(field: str, amount: int = 1):
    state = get_xpstate()
    state[field] = state.get(field, 0) + amount
    db["xpstate"].update_one({"_id": state["_id"]}, {"$set": {field: state[field]}})
    check_achievements()


def get_achievements() -> List[Dict[str, Any]]:
    return [serialize(a) for a in db["achievement"].find({})]


PREDEFINED_ACHIEVEMENTS = {
    "consistency_beast": {
        "name": "Consistency Beast",
        "condition": lambda s: s.get("streak", 0) >= 7,
    },
    "momentum_master": {
        "name": "Momentum Master",
        "condition": lambda s: s.get("streak", 0) >= 14,
    },
    "habit_hacker": {
        "name": "Habit Hacker",
        "condition": lambda s: s.get("habit_completions", 0) >= 30,
    },
    "emotional_clarity": {
        "name": "Emotional Clarity",
        "condition": lambda s: s.get("mood_logs", 0) >= 7,
    },
    "level_5": {
        "name": "Level 5 Achieved",
        "condition": lambda s: s.get("level", 1) >= 5,
    },
    "productivity_spike": {
        "name": "Productivity Spike",
        "condition": lambda s: s.get("missions_finished", 0) >= 5,
    },
}


def unlock_achievement(key: str, name: str):
    existing = db["achievement"].find_one({"key": key})
    if existing:
        return
    db["achievement"].insert_one(
        {"key": key, "name": name, "unlocked_at": datetime.now(timezone.utc).isoformat()}
    )


def check_achievements():
    s = get_xpstate()
    for key, spec in PREDEFINED_ACHIEVEMENTS.items():
        if spec["condition"](s):
            unlock_achievement(key, spec["name"])


# -------------------- Schemas Endpoint --------------------
@app.get("/schema")
def get_schema():
    return schemas.SCHEMA_MODELS


# -------------------- Habits --------------------
class HabitIn(BaseModel):
    name: str
    active: bool = True
    schedule: Optional[List[int]] = None


@app.get("/api/habits")
def list_habits():
    return [serialize(h) for h in db["habit"].find({})]


@app.post("/api/habits")
def create_habit(habit: HabitIn):
    hid = create_document("habit", habit.model_dump())
    return {"id": hid}


@app.put("/api/habits/{habit_id}")
def update_habit(habit_id: str, habit: HabitIn):
    res = db["habit"].update_one({"_id": to_object_id(habit_id)}, {"$set": habit.model_dump()})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Habit not found")
    return {"ok": True}


@app.delete("/api/habits/{habit_id}")
def delete_habit(habit_id: str):
    db["habit"].delete_one({"_id": to_object_id(habit_id)})
    return {"ok": True}


# Track habit completion for a given date (default today)
class HabitCheckIn(BaseModel):
    date: Optional[str] = None
    completed: bool = True


@app.post("/api/habits/{habit_id}/check")
def check_habit(habit_id: str, body: HabitCheckIn):
    d = (body.date or iso_today())
    habit = db["habit"].find_one({"_id": to_object_id(habit_id)})
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found")

    # upsert habitlog
    existing = db["habitlog"].find_one({"habit_id": habit_id, "date": d})
    xp_awarded = existing.get("xp_awarded", False) if existing else False
    if existing:
        db["habitlog"].update_one(
            {"_id": existing["_id"]}, {"$set": {"completed": body.completed}}
        )
    else:
        db["habitlog"].insert_one(
            {"habit_id": habit_id, "date": d, "completed": body.completed, "xp_awarded": False}
        )

    # XP: +10 once per habit per day when completed flips to True and not yet awarded
    if body.completed and not xp_awarded:
        db["habitlog"].update_one(
            {"habit_id": habit_id, "date": d}, {"$set": {"xp_awarded": True}}
        )
        add_xp(10)
        increment_counter("habit_completions", 1)

    return {"ok": True}


# -------------------- Mood --------------------
class MoodIn(BaseModel):
    date: Optional[str] = None
    rating: int


@app.post("/api/mood")
def log_mood(mood: MoodIn):
    d = (mood.date or iso_today())
    existing = db["mood"].find_one({"date": d})
    if existing:
        # Update rating, preserve xp_awarded flag
        db["mood"].update_one(
            {"_id": existing["_id"]}, {"$set": {"rating": mood.rating}}
        )
        awarded = existing.get("xp_awarded", False)
    else:
        db["mood"].insert_one({"date": d, "rating": mood.rating, "xp_awarded": False})
        awarded = False

    if not awarded:
        db["mood"].update_one({"date": d}, {"$set": {"xp_awarded": True}})
        add_xp(5)
        increment_counter("mood_logs", 1)

    return {"ok": True}


@app.get("/api/mood/{d}")
def get_mood(d: str):
    m = db["mood"].find_one({"date": d})
    return serialize(m) if m else None


# -------------------- Tasks --------------------
class TaskIn(BaseModel):
    title: str
    date: Optional[str] = None
    status: str = "open"  # open|completed|deferred|archived
    note: Optional[str] = None


@app.get("/api/tasks")
def list_tasks(d: Optional[str] = Query(default=None)):
    filt = {"date": d or iso_today()} if d or d is None else {}
    tasks = [serialize(t) for t in db["task"].find(filt)]
    return tasks


@app.post("/api/tasks")
def create_task(task: TaskIn):
    payload = task.model_dump()
    if not payload.get("date"):
        payload["date"] = iso_today()
    tid = create_document("task", payload)
    return {"id": tid}


@app.put("/api/tasks/{task_id}")
def update_task(task_id: str, task: TaskIn):
    res = db["task"].update_one({"_id": to_object_id(task_id)}, {"$set": task.model_dump()})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True}


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    db["task"].delete_one({"_id": to_object_id(task_id)})
    return {"ok": True}


# -------------------- Mission of the Day --------------------
class MissionIn(BaseModel):
    date: Optional[str] = None
    text: str
    done: bool = False


@app.get("/api/mission")
def get_mission(d: Optional[str] = Query(default=None)):
    d = d or iso_today()
    m = db["mission"].find_one({"date": d})
    return serialize(m) if m else None


@app.post("/api/mission")
def set_mission(m: MissionIn):
    d = m.date or iso_today()
    existing = db["mission"].find_one({"date": d})
    payload = {"date": d, "text": m.text, "done": m.done, "xp_awarded": False}
    if existing:
        db["mission"].update_one({"_id": existing["_id"]}, {"$set": payload})
        doc = db["mission"].find_one({"_id": existing["_id"]})
    else:
        db["mission"].insert_one(payload)
        doc = db["mission"].find_one({"date": d})
    return serialize(doc)


class MissionDoneIn(BaseModel):
    date: Optional[str] = None
    done: bool = True


@app.post("/api/mission/done")
def complete_mission(body: MissionDoneIn):
    d = body.date or iso_today()
    m = db["mission"].find_one({"date": d})
    if not m:
        raise HTTPException(status_code=404, detail="Mission not set for this day")
    db["mission"].update_one({"_id": m["_id"]}, {"$set": {"done": body.done}})
    if body.done and not m.get("xp_awarded", False):
        db["mission"].update_one({"_id": m["_id"]}, {"$set": {"xp_awarded": True}})
        add_xp(15)
        increment_counter("missions_finished", 1)
    return {"ok": True}


# -------------------- Notes --------------------
class NoteIn(BaseModel):
    title: str
    text: str = ""
    category: Optional[str] = None


@app.get("/api/notes")
def list_notes(category: Optional[str] = None):
    filt = {"category": category} if category else {}
    return [serialize(n) for n in db["note"].find(filt)]


@app.post("/api/notes")
def create_note(note: NoteIn):
    nid = create_document("note", note.model_dump())
    return {"id": nid}


@app.put("/api/notes/{note_id}")
def update_note(note_id: str, note: NoteIn):
    res = db["note"].update_one({"_id": to_object_id(note_id)}, {"$set": note.model_dump()})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"ok": True}


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: str):
    db["note"].delete_one({"_id": to_object_id(note_id)})
    return {"ok": True}


# -------------------- Daily Summary & Completion --------------------

def weekday_index(iso_d: str) -> int:
    return datetime.fromisoformat(iso_d).weekday()


def is_habit_scheduled(h: Dict[str, Any], d: str) -> bool:
    sched = h.get("schedule")
    if not sched:
        return True
    return weekday_index(d) in sched


def evaluate_day(d: str) -> Dict[str, Any]:
    # Habits tracked: all scheduled habits must have habitlog with completed True
    habits = list(db["habit"].find({"active": True}))
    scheduled = [h for h in habits if is_habit_scheduled(h, d)]
    all_habits_done = True
    for h in scheduled:
        log = db["habitlog"].find_one({"habit_id": str(h["_id"]), "date": d})
        if not log or not log.get("completed", False):
            all_habits_done = False
            break

    # Mood logged
    mood = db["mood"].find_one({"date": d})
    mood_logged = bool(mood)

    # Tasks updated: no tasks with status 'open' for the day
    open_count = db["task"].count_documents({"date": d, "status": "open"})
    tasks_updated = open_count == 0

    # Mission done?
    mission = db["mission"].find_one({"date": d})

    return {
        "habits_done": all_habits_done,
        "mood_logged": mood_logged,
        "tasks_updated": tasks_updated,
        "mission_done": bool(mission and mission.get("done", False)),
    }


class CompleteDayIn(BaseModel):
    date: Optional[str] = None


@app.post("/api/day/complete")
def complete_day(body: CompleteDayIn):
    d = body.date or iso_today()
    evalr = evaluate_day(d)
    complete = evalr["habits_done"] and evalr["mood_logged"] and evalr["tasks_updated"]

    # Upsert dailysummary
    ds = db["dailysummary"].find_one({"date": d})
    if ds:
        db["dailysummary"].update_one(
            {"_id": ds["_id"]},
            {"$set": {
                "habits_done": evalr["habits_done"],
                "mood_logged": evalr["mood_logged"],
                "tasks_updated": evalr["tasks_updated"],
                "complete": complete,
            }},
        )
        ds = db["dailysummary"].find_one({"_id": ds["_id"]})
    else:
        db["dailysummary"].insert_one(
            {
                "date": d,
                "habits_done": evalr["habits_done"],
                "mood_logged": evalr["mood_logged"],
                "tasks_updated": evalr["tasks_updated"],
                "complete": complete,
                "xp_awarded": False,
                "streak_after": 0,
            }
        )
        ds = db["dailysummary"].find_one({"date": d})

    # Award XP for daily completion once
    if complete and not ds.get("xp_awarded", False):
        add_xp(25)
        db["dailysummary"].update_one({"_id": ds["_id"]}, {"$set": {"xp_awarded": True}})

    # Streak calculation: if most recent previous complete date is yesterday, continue; else reset->1
    y = (datetime.fromisoformat(d).date() - timedelta(days=1)).isoformat()
    prev_complete = db["dailysummary"].find_one({"date": y, "complete": True})
    if complete:
        if prev_complete:
            new_streak = get_xpstate().get("streak", 0) + 1
        else:
            new_streak = 1
        set_streak(new_streak)
        db["dailysummary"].update_one({"_id": ds["_id"]}, {"$set": {"streak_after": new_streak}})
    return {"complete": complete, "evaluation": evalr}


@app.get("/api/day/status")
def day_status(d: Optional[str] = Query(default=None)):
    d = d or iso_today()
    e = evaluate_day(d)
    ds = db["dailysummary"].find_one({"date": d})
    return {"date": d, "evaluation": e, "summary": serialize(ds) if ds else None, "xpstate": serialize(get_xpstate())}


# -------------------- Weekly Overview --------------------

def week_bounds(any_day: str) -> (str, str):
    dt = datetime.fromisoformat(any_day).date()
    # ISO Monday start
    start = dt - timedelta(days=dt.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


@app.get("/api/weekly")
def weekly_overview(week_start: Optional[str] = Query(default=None)):
    d = week_start or iso_today()
    ws, we = week_bounds(d)

    # Habit completion pct: total scheduled habit instances vs completed
    habits = list(db["habit"].find({"active": True}))
    days = [(datetime.fromisoformat(ws) + timedelta(days=i)).date().isoformat() for i in range(7)]
    scheduled_count = 0
    completed_count = 0
    for day in days:
        for h in habits:
            if is_habit_scheduled(h, day):
                scheduled_count += 1
                log = db["habitlog"].find_one({"habit_id": str(h["_id"]), "date": day})
                if log and log.get("completed", False):
                    completed_count += 1
    habit_pct = (completed_count / scheduled_count * 100.0) if scheduled_count else 0.0

    # Mood average
    moods = list(db["mood"].find({"date": {"$gte": ws, "$lte": we}}))
    mood_avg = (sum(m.get("rating", 0) for m in moods) / len(moods)) if moods else None

    # Task completion %
    total_tasks = db["task"].count_documents({"date": {"$gte": ws, "$lte": we}})
    completed_tasks = db["task"].count_documents({"date": {"$gte": ws, "$lte": we}, "status": "completed"})
    task_pct = (completed_tasks / total_tasks * 100.0) if total_tasks else 0.0

    # XP earned: from awarded flags
    xp = 0
    xp += db["habitlog"].count_documents({"date": {"$gte": ws, "$lte": we}, "xp_awarded": True}) * 10
    xp += db["mood"].count_documents({"date": {"$gte": ws, "$lte": we}, "xp_awarded": True}) * 5
    xp += db["mission"].count_documents({"date": {"$gte": ws, "$lte": we}, "xp_awarded": True}) * 15
    xp += db["dailysummary"].count_documents({"date": {"$gte": ws, "$lte": we}, "xp_awarded": True}) * 25

    # Streak progression
    dss = list(db["dailysummary"].find({"date": {"$gte": ws, "$lte": we}}).sort("date", 1))
    streak_start = dss[0].get("streak_after", 0) if dss else get_xpstate().get("streak", 0)
    streak_end = dss[-1].get("streak_after", 0) if dss else get_xpstate().get("streak", 0)

    highlights = {
        "best_mood_day": max(moods, key=lambda m: m.get("rating", 0))["date"] if moods else None,
        "most_tasks_done_day": None,
    }
    # Most tasks done day
    if total_tasks:
        counts: Dict[str, int] = {}
        for t in db["task"].find({"date": {"$gte": ws, "$lte": we}, "status": "completed"}):
            counts[t["date"]] = counts.get(t["date"], 0) + 1
        if counts:
            highlights["most_tasks_done_day"] = max(counts.items(), key=lambda kv: kv[1])[0]

    # Upsert weeklysummary
    existing = db["weeklysummary"].find_one({"week_start": ws})
    payload = {
        "week_start": ws,
        "week_end": we,
        "habit_completion_pct": habit_pct,
        "mood_avg": mood_avg,
        "task_completion_pct": task_pct,
        "xp_earned": xp,
        "streak_start": streak_start,
        "streak_end": streak_end,
        "highlights": highlights,
    }
    if existing:
        db["weeklysummary"].update_one({"_id": existing["_id"]}, {"$set": payload})
        doc = db["weeklysummary"].find_one({"_id": existing["_id"]})
    else:
        payload["bonus_awarded"] = False
        db["weeklysummary"].insert_one(payload)
        doc = db["weeklysummary"].find_one({"week_start": ws})

    return serialize(doc)


class WeeklyBonusIn(BaseModel):
    week_start: Optional[str] = None


@app.post("/api/weekly/bonus")
def weekly_bonus(body: WeeklyBonusIn):
    ws, we = week_bounds(body.week_start or iso_today())
    doc = db["weeklysummary"].find_one({"week_start": ws})
    if not doc:
        raise HTTPException(status_code=404, detail="Weekly summary not found. Call /api/weekly first.")
    if doc.get("bonus_awarded", False):
        return {"ok": True, "already_awarded": True}
    add_xp(50)
    db["weeklysummary"].update_one({"_id": doc["_id"]}, {"$set": {"bonus_awarded": True}})
    return {"ok": True}


# -------------------- Achievements & XP State --------------------
@app.get("/api/xpstate")
def xpstate():
    return serialize(get_xpstate())


@app.get("/api/achievements")
def achievements():
    return get_achievements()


# -------------------- Health --------------------
@app.get("/")
def read_root():
    return {"message": "Sola API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
