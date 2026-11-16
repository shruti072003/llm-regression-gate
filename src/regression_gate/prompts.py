"""Load and validate versioned prompt files from prompts/."""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from regression_gate.schemas import EmailClassification

PROMPTS_DIR = Path("prompts")


class PromptConfigError(ValueError):
    """A prompt file is missing, unreadable, or invalid."""


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    email: str = Field(min_length=1)
    # Validated with the same schema as model answers, so an example can
    # never teach a format the validator would reject.
    output: EmailClassification


class PromptConfig(BaseModel):
    # forbid: a typo like "temprature" fails loudly instead of being dropped.
    # frozen: nothing can change the settings in the middle of a run.
    model_config = ConfigDict(extra="forbid", frozen=True)

    # No defaults: the file must be the complete record of what ran.
    version_id: str = Field(pattern=r"^v\d+$")
    created_at: date
    description: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    system_prompt: str = Field(min_length=1)
    # A tuple, not a list: frozen only blocks reassigning fields, so a list
    # could still be appended to.
    few_shot: tuple[FewShotExample, ...]


def prompt_path(version_id: str, prompts_dir: Path = PROMPTS_DIR) -> Path:
    return prompts_dir / f"{version_id}.yaml"


def load_prompt(path: Path) -> PromptConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as e:
        raise PromptConfigError(f"{path}: cannot read file ({e})") from e
    except yaml.YAMLError as e:
        raise PromptConfigError(f"{path}: invalid YAML ({e})") from e

    if not isinstance(raw, dict):
        raise PromptConfigError(f"{path}: expected a mapping of fields at the top level")

    try:
        config = PromptConfig.model_validate(raw)
    except ValidationError as e:
        raise PromptConfigError(f"{path}: {e}") from e

    # Catches copying v1.yaml to v2.yaml and forgetting to update version_id,
    # which would store every result under the wrong version.
    if config.version_id != path.stem:
        raise PromptConfigError(
            f"{path}: version_id is {config.version_id!r} but the filename says {path.stem!r}"
        )

    return config
