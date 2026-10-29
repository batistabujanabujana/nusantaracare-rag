from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    confidence_label: Literal["high", "medium", "low"]
    reason_code: Literal[
        "answered", "no_relevant_context", "out_of_scope", "unsafe_request",
        "service_unavailable", "invalid_request"
    ]
