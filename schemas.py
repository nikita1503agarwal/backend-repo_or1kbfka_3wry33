"""
Database Schemas for Sola

Each Pydantic model maps to a MongoDB collection (lowercase of class name).
These schemas are used for validation and for the /schema endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Core entities
class Habit(BaseModel):
    name: str = Field(..., description="Habit display name")
    active: bool = Field(True, description="Whether the habit is active")
    schedule: Optional[List[int]] = Field(
        default=None,
        description="Optional list of weekday indexes (0=Mon..6=Sun) when habit is scheduled. None = daily.",
    )

class Habitlog(BaseModel):
    habit_id: str = Field(..., description="Reference to habit _id as string")
    date: str = Field(..., description="ISO date YYYY-MM-DD")
    completed: bool = Field(False, description="Completed status for this date")
    xp_awarded: bool = Field(False, description="Whether XP was already awarded for this completion")

class Mood(BaseModel):
    date: str = Field(..., description="ISO date YYYY-MM-DD")
    rating: int = Field(..., ge=1, le=5, description="Mood rating 1-5")
    xp_awarded: bool = Field(False, description="Whether XP was awarded for mood log")

class Task(BaseModel):
    title: str = Field(..., description="Task title")
    date: str = Field(..., description="ISO date YYYY-MM-DD the task belongs to")
    status: str = Field("open", description="open|completed|deferred|archived")
    note: Optional[str] = Field(None, description="Optional notes")

class Mission(BaseModel):
    date: str = Field(..., description="ISO date YYYY-MM-DD")
    text: str = Field(..., description="Mission of the day text")
    done: bool = Field(False, description="Whether the mission is completed")
    xp_awarded: bool = Field(False, description="Whether XP was awarded for mission completion")

class Dailysummary(BaseModel):
    date: str = Field(..., description="ISO date YYYY-MM-DD")
    habits_done: bool = Field(False)
    mood_logged: bool = Field(False)
    tasks_updated: bool = Field(False)
    complete: bool = Field(False)
    xp_awarded: bool = Field(False, description="Whether +25 XP for daily completion was awarded")
    streak_after: int = Field(0, description="Streak after marking complete")

class Xpstate(BaseModel):
    total_xp: int = Field(0)
    level: int = Field(1)
    xp_in_level: int = Field(0, description="XP progress within current level")
    xp_for_next: int = Field(100, description="XP needed to reach next level from start of level")
    streak: int = Field(0)
    missions_finished: int = Field(0)
    mood_logs: int = Field(0)
    habit_completions: int = Field(0)

class Achievement(BaseModel):
    key: str = Field(..., description="Unique achievement key")
    name: str = Field(..., description="Display name")
    unlocked_at: Optional[str] = Field(None, description="ISO datetime")

class Weeklysummary(BaseModel):
    week_start: str = Field(..., description="ISO date YYYY-MM-DD (Monday)")
    week_end: str = Field(..., description="ISO date YYYY-MM-DD (Sunday)")
    habit_completion_pct: float = Field(0)
    mood_avg: Optional[float] = Field(None)
    task_completion_pct: float = Field(0)
    xp_earned: int = Field(0)
    streak_start: int = Field(0)
    streak_end: int = Field(0)
    highlights: Optional[Dict[str, Any]] = Field(default=None)
    bonus_awarded: bool = Field(False)

class Note(BaseModel):
    title: str = Field(...)
    text: str = Field("")
    category: Optional[str] = Field(None, description="Category label")

# Export schema metadata for /schema endpoint consumers
SCHEMA_MODELS = {
    "habit": Habit.model_json_schema(),
    "habitlog": Habitlog.model_json_schema(),
    "mood": Mood.model_json_schema(),
    "task": Task.model_json_schema(),
    "mission": Mission.model_json_schema(),
    "dailysummary": Dailysummary.model_json_schema(),
    "xpstate": Xpstate.model_json_schema(),
    "achievement": Achievement.model_json_schema(),
    "weeklysummary": Weeklysummary.model_json_schema(),
    "note": Note.model_json_schema(),
}
