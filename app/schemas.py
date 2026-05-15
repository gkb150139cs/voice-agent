from typing import Literal, Optional

from pydantic import BaseModel, Field


PredictionLabel = Literal["male", "female", "unknown"]
AgeBracket = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQuality = Literal["good", "degraded", "insufficient"]


class AttributePrediction(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class LanguageGuess(BaseModel):
    prediction: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: AttributePrediction
    age_bracket: AttributePrediction
    processing_ms: int
    audio_quality: AudioQuality
    language: Optional[LanguageGuess] = None


class StreamPartialResponse(BaseModel):
    contact_id: str
    gender: AttributePrediction
    age_bracket: AttributePrediction
    audio_quality: AudioQuality
    window_seconds: float
