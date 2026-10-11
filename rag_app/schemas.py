from typing import Literal
from pydantic import BaseModel, Field, field_validator

Strategy = Literal['simple', 'crag', 'multi_step']


class RouteDecision(BaseModel):
    strategy: Strategy
    standalone_question: str = Field(min_length=1, max_length=3000)
    reason: str = Field(max_length=1200)
    clarification: str = Field(default='', max_length=1000,
        description='A short clarifying question for greetings, out-of-scope requests, or unresolved pronouns; empty otherwise.')


class DocumentGrade(BaseModel):
    relevant_ids: list[str] = Field(max_length=8)
    sufficient: bool
    reason: str = Field(max_length=1200)


class Rewrite(BaseModel):
    query: str = Field(min_length=1, max_length=3000)


class Plan(BaseModel):
    subquestions: list[str] = Field(min_length=2, max_length=4)

    @field_validator('subquestions')
    @classmethod
    def bounded_questions(cls, values):
        if any(not value.strip() or len(value) > 1500 for value in values):
            raise ValueError('Each subquestion must be nonblank and at most 1500 characters.')
        if len({value.strip().lower() for value in values}) != len(values):
            raise ValueError('Use distinct subquestions, not duplicate paraphrases.')
        return [value.strip() for value in values]


class Draft(BaseModel):
    answerable: bool
    answer: str = Field(max_length=6000)
    citations: list[str] = Field(max_length=16, description='Supporting supplied passage IDs, for example S1 and S2.')


class AnswerCheck(BaseModel):
    grounded: bool
    complete: bool
    reason: str = Field(max_length=1200)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    session_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    request_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    source: str | None = Field(default=None, max_length=150)

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Enter a question.')
        return value.strip()


class Source(BaseModel):
    id: str
    source: str
    chunk_id: str
    excerpt: str


class Response(BaseModel):
    answer: str
    status: Literal['answered', 'insufficient_evidence', 'clarification_needed']
    initial_strategy: Strategy
    final_strategy: Strategy
    route_reason: str
    standalone_question: str
    sources: list[Source]
    events: list[dict]
    subquestions: list[str]
    retrieval_rounds: int
    correction_attempts: int
    trace_id: str | None
    request_id: str
    session_id: str
    latency_ms: int
    usage: dict
