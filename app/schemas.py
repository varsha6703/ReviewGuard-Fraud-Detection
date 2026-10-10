
from pydantic import BaseModel, Field, ConfigDict


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(
        ...,
        gt=0,
        description="Unique ID of the user submitting the review"
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Review text to analyze"
    )

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Product rating between 1 and 5"
    )

    velocity: int = Field(
        default=0,
        ge=0,
        description="Number of reviews posted by the user in the last 24 hours"
    )
