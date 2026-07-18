"""Tests for checkpointed/resumable synthetic data generation (network-free: ScriptedLLMClient)."""

import json

from src.data.synthetic.generate import _count_existing_samples_per_label, generate_dataset
from src.data.synthetic.llm_client import ScriptedLLMClient
from src.data.synthetic.schema import SyntheticSample, now_iso
from src.utils.emotion_constants import DEFAULT_EMOTION_LABELS

_DATAGEN_CONFIG_YAML = """
seed: 1
generation:
  target_per_label: 2
  max_new_tokens: 50
  temperature: 0.5
  top_p: 0.9
  log_every_n_samples: 100
diversity_axes:
  van_phong: ["style_a"]
  do_dai: ["short"]
  ngu_canh: ["context_a"]
"""


def _write_config(tmp_path):
    config_file = tmp_path / "datagen_config.yaml"
    config_file.write_text(_DATAGEN_CONFIG_YAML, encoding="utf-8")
    return config_file


def test_generate_dataset_writes_target_per_label_samples_for_every_label(tmp_path):
    config_file = _write_config(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    client = ScriptedLLMClient(responses=["mẫu nhật ký test"])

    generate_dataset(
        client=client,
        model_display_name="Scripted-Test",
        channel="scripted",
        output_path=str(output_path),
        generation_round=1,
        config_path=str(config_file),
    )

    counts = _count_existing_samples_per_label(output_path)
    assert counts == dict.fromkeys(DEFAULT_EMOTION_LABELS, 2)


def test_generate_dataset_writes_full_provenance(tmp_path):
    config_file = _write_config(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    client = ScriptedLLMClient(responses=["mẫu nhật ký test"])

    generate_dataset(
        client=client,
        model_display_name="Scripted-Test",
        channel="scripted",
        output_path=str(output_path),
        generation_round=3,
        config_path=str(config_file),
    )

    with open(output_path, encoding="utf-8") as f:
        first_row = json.loads(f.readline())
    sample = SyntheticSample(**first_row)

    assert sample.model == "Scripted-Test"
    assert sample.channel == "scripted"
    assert sample.generation_round == 3
    assert sample.axis_style == "style_a"
    assert sample.axis_length == "short"
    assert sample.axis_context == "context_a"


def test_generate_dataset_resumes_without_duplicating_existing_samples(tmp_path):
    config_file = _write_config(tmp_path)
    output_path = tmp_path / "raw.jsonl"

    existing_sample = SyntheticSample(
        sample_id="scripted-test-0-preexisting",
        text="mẫu đã có sẵn từ trước",
        label=0,
        label_name="Enjoyment",
        model="Scripted-Test",
        channel="scripted",
        axis_style="style_a",
        axis_length="short",
        axis_context="context_a",
        prompt_template_id="diary_v1",
        generation_round=1,
        generated_at=now_iso(),
    )
    output_path.write_text(existing_sample.model_dump_json() + "\n", encoding="utf-8")

    client = ScriptedLLMClient(responses=["mẫu mới"])
    generate_dataset(
        client=client,
        model_display_name="Scripted-Test",
        channel="scripted",
        output_path=str(output_path),
        generation_round=1,
        config_path=str(config_file),
    )

    counts = _count_existing_samples_per_label(output_path)
    assert counts[0] == 2  # 1 đã có sẵn + 1 sinh thêm (target_per_label=2), không sinh thừa

    with open(output_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["sample_id"] == "scripted-test-0-preexisting"


def test_generate_dataset_skips_sample_on_client_error_and_continues(tmp_path, caplog):
    config_file = _write_config(tmp_path)
    output_path = tmp_path / "raw.jsonl"

    class _FailingClient:
        def generate(self, prompt, max_tokens=220, temperature=0.9, top_p=0.95):
            raise RuntimeError("simulated LLM failure")

    with caplog.at_level("ERROR"):
        generate_dataset(
            client=_FailingClient(),
            model_display_name="Scripted-Test",
            channel="scripted",
            output_path=str(output_path),
            generation_round=1,
            config_path=str(config_file),
        )

    assert output_path.read_text(encoding="utf-8") == ""
    assert "Sinh mẫu lỗi" in caplog.text
