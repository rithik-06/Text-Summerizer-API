from pydantic import BaseModel, Field, validator
from typing import Optional, Literal


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=10000)
    max_length: Optional[int] = Field(130, ge=30, le=500)
    min_length: Optional[int] = Field(30, ge=10, le=100)
    summary_type: Optional[Literal["short", "medium", "detailed"]] = "medium"
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Your long text here to be summarized...",
                "max_length": 130,
                "min_length": 30,
                "summary_type": "medium"
            }
        }


class SummarizeResponse(BaseModel):
    original_length: int
    summary: str
    summary_length: int
    compression_ratio: float