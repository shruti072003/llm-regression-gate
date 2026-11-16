from pathlib import Path

import pytest
import yaml

from regression_gate.prompts import PromptConfigError, load_prompt, prompt_path
from regression_gate.schemas import Category

VALID = {
    "version_id": "v7",
    "created_at": "2026-11-15",
    "description": "Test prompt.",
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "system_prompt": "Classify the email.",
    "few_shot": [
        {"email": "I was charged twice.", "output": {"category": "billing", "summary": "Duplicate charge."}}
    ],
}


def write_prompt(tmp_path: Path, data: dict, name: str = "v7") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_real_v1_prompt_loads():
    config = load_prompt(prompt_path("v1"))
    assert config.version_id == "v1"
    assert config.few_shot == ()


def test_valid_prompt_loads(tmp_path):
    config = load_prompt(write_prompt(tmp_path, VALID))
    assert config.temperature == 0
    assert config.few_shot[0].output.category is Category.BILLING


def test_typo_field_is_rejected(tmp_path):
    data = {**VALID, "temprature": 0.7}
    with pytest.raises(PromptConfigError, match="temprature"):
        load_prompt(write_prompt(tmp_path, data))


def test_missing_field_is_rejected(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "temperature"}
    with pytest.raises(PromptConfigError, match="temperature"):
        load_prompt(write_prompt(tmp_path, data))


def test_version_must_match_filename(tmp_path):
    with pytest.raises(PromptConfigError, match="filename"):
        load_prompt(write_prompt(tmp_path, VALID, name="v8"))


@pytest.mark.parametrize("version_id", ["1.0.0", "V1", "v", "version1"])
def test_version_format_is_enforced(tmp_path, version_id):
    data = {**VALID, "version_id": version_id}
    with pytest.raises(PromptConfigError):
        load_prompt(write_prompt(tmp_path, data, name=version_id))


@pytest.mark.parametrize("temperature", [-0.1, 2.5])
def test_temperature_out_of_range_is_rejected(tmp_path, temperature):
    data = {**VALID, "temperature": temperature}
    with pytest.raises(PromptConfigError, match="temperature"):
        load_prompt(write_prompt(tmp_path, data))


def test_few_shot_with_invalid_category_is_rejected(tmp_path):
    data = {**VALID, "few_shot": [{"email": "Hi", "output": {"category": "payments", "summary": "Something."}}]}
    with pytest.raises(PromptConfigError, match="category"):
        load_prompt(write_prompt(tmp_path, data))


def test_config_cannot_be_changed_after_loading(tmp_path):
    config = load_prompt(write_prompt(tmp_path, VALID))
    with pytest.raises(Exception):
        config.temperature = 1.0


def test_invalid_yaml_is_reported(tmp_path):
    path = tmp_path / "v7.yaml"
    path.write_text("few_shot\n")
    with pytest.raises(PromptConfigError):
        load_prompt(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(PromptConfigError, match="cannot read"):
        load_prompt(tmp_path / "v99.yaml")
