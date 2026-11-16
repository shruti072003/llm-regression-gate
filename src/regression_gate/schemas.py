from enum import StrEnum
from pydantic import BaseModel, Field, ConfigDict, field_validator

class Category(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"

class EmailClassification(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    category: Category
    summary: str = Field(min_length=3, max_length=100)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value):
        # Runs before the Enum check, so " Billing " is accepted as "billing".
        # The classifier should still record when this happened (format compliance metric).
        if isinstance(value, str):
            return value.strip().lower()
        return value
