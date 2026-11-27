"""Load gate.yaml: thresholds, judge model, pricing."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SETTINGS_PATH = Path("gate.yaml")

Fraction = Annotated[float, Field(ge=0, le=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JudgeSettings(_Strict):
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)


class Thresholds(_Strict):
    accuracy_warn_drop: Fraction
    accuracy_critical_drop: Fraction
    category_warn_drop: Fraction
    category_critical_drop: Fraction
    summary_warn_drop: Fraction
    summary_critical_drop: Fraction
    max_infra_error_rate: Fraction


class DriftSettings(_Strict):
    window: int = Field(ge=2)
    accuracy_warn_drop: Fraction
    summary_warn_drop: Fraction


class Price(_Strict):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class GateSettings(_Strict):
    active_prompt: str = Field(pattern=r"^v\d+$")
    dataset: Path
    concurrency: int = Field(ge=1, le=64)
    judge: JudgeSettings
    thresholds: Thresholds
    drift: DriftSettings
    pricing: dict[str, Price]

    def price_for(self, served_model: str | None) -> Price | None:
        """Longest configured name that the served snapshot starts with, so
        gpt-4.1-mini-2025-04-14 matches gpt-4.1-mini, not gpt-4.1."""
        if not served_model:
            return None
        matches = [name for name in self.pricing if served_model.startswith(name)]
        return self.pricing[max(matches, key=len)] if matches else None

    def cost(self, served_model: str | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
        price = self.price_for(served_model)
        if price is None or input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def load_settings(path: Path = SETTINGS_PATH) -> GateSettings:
    try:
        raw = yaml.safe_load(Path(path).read_text())
        return GateSettings.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as e:
        raise ValueError(f"{path}: {e}") from e
