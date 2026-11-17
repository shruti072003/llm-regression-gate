"""Load and validate the human-written golden dataset (data/golden.yaml)."""

import hashlib
import re
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from regression_gate.schemas import Category

GOLDEN_PATH = Path("data/golden.yaml")


class DatasetError(ValueError):
    """The golden dataset is missing, unreadable, or invalid."""


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EdgeTag(StrEnum):
    # A closed set, like Category: a typo such as "sarcastic" fails instead of
    # creating a separate group in the per-tag report.
    AMBIGUOUS = "ambiguous"  # reasonable people could pick different categories
    POLICY_BOUNDARY = "policy_boundary"  # a written labeling rule decides it (e.g. login error page)
    MULTI_TOPIC = "multi_topic"
    VERY_SHORT = "very_short"
    TYPOS = "typos"
    MIXED_LANGUAGE = "mixed_language"
    SARCASM = "sarcasm"


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Neutral and permanent: never encodes the label, never reused.
    id: str = Field(pattern=r"^case-\d{3}$")
    email: str = Field(min_length=1)
    # Strict, unlike model output: "Billing" here is a typo in my own data.
    expected_category: Category
    # What a correct summary must mention; the Step 4 judge checks each one.
    key_facts: tuple[str, ...] = Field(min_length=1, max_length=3)
    difficulty: Difficulty
    tags: tuple[EdgeTag, ...]
    notes: str | None = None

    @field_validator("key_facts")
    @classmethod
    def facts_not_blank(cls, facts):
        if any(not f.strip() for f in facts):
            raise ValueError("key facts must not be blank")
        return facts

    @field_validator("tags")
    @classmethod
    def tags_unique(cls, tags):
        if len(set(tags)) != len(tags):
            raise ValueError("tags must not repeat")
        return tags

    @model_validator(mode="after")
    def notes_required_for_tricky_cases(self):
        # Easy cases explain themselves; hard or tagged ones need the reasoning written down.
        if (self.difficulty is Difficulty.HARD or self.tags) and not (self.notes and self.notes.strip()):
            raise ValueError(f"{self.id}: hard or tagged cases need notes explaining the label")
        return self


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump when a case is added, removed or relabeled. content_hash catches a forgotten bump.
    version: int = Field(ge=1)
    description: str = Field(min_length=1)
    cases: tuple[GoldenCase, ...]
    # Set by load_dataset from the file bytes, not written in the file.
    content_hash: str = ""

    @model_validator(mode="after")
    def no_duplicates(self):
        seen_ids, seen_emails = set(), {}
        for case in self.cases:
            if case.id in seen_ids:
                raise ValueError(f"duplicate id {case.id}")
            seen_ids.add(case.id)
            key = normalize_email(case.email)
            if key in seen_emails:
                raise ValueError(f"{case.id} has the same email as {seen_emails[key]}")
            seen_emails[key] = case.id
        return self


def normalize_email(text: str) -> str:
    """Lowercase and collapse whitespace, so trivially reformatted copies still match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_dataset(path: Path = GOLDEN_PATH) -> GoldenDataset:
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
    except OSError as e:
        raise DatasetError(f"{path}: cannot read file ({e})") from e
    except yaml.YAMLError as e:
        raise DatasetError(f"{path}: invalid YAML ({e})") from e

    if not isinstance(raw, dict):
        raise DatasetError(f"{path}: expected a mapping of fields at the top level")
    if "content_hash" in raw:
        raise DatasetError(f"{path}: content_hash is computed, not written in the file")

    try:
        return GoldenDataset.model_validate({**raw, "content_hash": hashlib.sha256(raw_bytes).hexdigest()[:12]})
    except ValidationError as e:
        raise DatasetError(f"{path}: {e}") from e


def find_leaks(dataset: GoldenDataset, other_emails: dict[str, str]) -> list[str]:
    """Golden cases whose email also appears elsewhere (few-shot examples, LLM-written
    smoke emails). `other_emails` maps each email to where it came from."""
    others = {normalize_email(email): source for email, source in other_emails.items()}
    return [
        f"{case.id} duplicates an email from {others[key]}"
        for case in dataset.cases
        if (key := normalize_email(case.email)) in others
    ]
