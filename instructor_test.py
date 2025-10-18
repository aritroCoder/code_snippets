from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List
from datetime import datetime, timezone
import instructor

OPENROUTER_API_KEY = (
    "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"
)

client = instructor.from_provider(
    "openrouter/google/gemma-3-27b-it:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    mode=instructor.Mode.JSON,
)


class Task(BaseModel):
    title: str = Field(..., min_length=3)
    due_date: datetime
    completed: bool = False

    @model_validator(mode="after")
    def validate_due_date(self):
        # Use timezone-aware datetime for comparison
        now = datetime.now(timezone.utc)
        # Ensure due_date is timezone-aware
        due_date_aware = self.due_date if self.due_date.tzinfo else self.due_date.replace(tzinfo=timezone.utc)
        if self.completed and due_date_aware > now:
            raise ValueError("Task cannot be marked complete before its due date.")
        return self


class Project(BaseModel):
    name: str = Field(..., min_length=3)
    tasks: List[Task]

    @field_validator("tasks")
    @classmethod
    def validate_task_count(cls, v):
        if len(v) == 0:
            raise ValueError("Project must contain at least one task.")
        return v


class Workspace(BaseModel):
    workspace_id: str = Field(..., pattern=r"^WS-\d{4}$")
    owner: str = Field(..., min_length=3)
    projects: List[Project]

    @model_validator(mode="after")
    def validate_workspace(self):
        if sum(len(p.tasks) for p in self.projects) < 2:
            raise ValueError("Workspace must have at least two tasks total.")
        return self


SAMPLE_TEXT = """
You are a precise JSON generator.

Produce a JSON object matching the following schema:
Workspace -> Projects -> Tasks (3 levels deep).
Each field must satisfy all validation rules.

Rules:
- workspace_id: format 'WS-XXXX' (digits only)
- owner: non-empty string
- each project must have ≥1 task
- total tasks across all projects ≥2
- if a task is completed, its due_date must not be in the future.

Example instruction:
"Generate a realistic example workspace for a software team with 2 projects and 3 tasks total."
"""


def call_structured_model(messages, response_model):
    response = client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        extra_body={"provider": {"require_parameters": True}},
        max_retries=3,
    )
    return response


if __name__ == "__main__":
    response: Workspace = call_structured_model(
        messages=[
            {
                "role": "system",
                "content": "You are a game master assistant that organizes character information clearly.",
            },
            {
                "role": "user",
                "content": f"Extract the character information from the following game text:\n\n{SAMPLE_TEXT}",
            },
        ],
        response_model=Workspace,
    )  # type: ignore

    print("\nFull JSON:")
    print(response.model_dump_json(indent=2))
