"""Tests for prompt composition (no I/O, no network)."""

import random

import pytest

from src.data.synthetic.prompts import EXAMPLE_BY_LABEL, build_prompt, sample_axis_values


def test_build_prompt_embeds_label_name_and_axis_values():
    prompt = build_prompt(
        label=1,
        style="hài hước, tự trào",
        length="ngắn (2-3 câu)",
        context="gia đình",
    )

    assert "Sadness" in prompt
    assert "hài hước, tự trào" in prompt
    assert "ngắn (2-3 câu)" in prompt
    assert "gia đình" in prompt
    assert EXAMPLE_BY_LABEL["Sadness"] in prompt


def test_build_prompt_raises_on_unknown_label():
    with pytest.raises(ValueError, match="Unknown label index"):
        build_prompt(label=99, style="a", length="b", context="c")


def test_sample_axis_values_is_deterministic_with_seeded_rng():
    axis_pools = {
        "van_phong": ["style_a", "style_b"],
        "do_dai": ["short", "long"],
        "ngu_canh": ["work", "family"],
    }

    first = sample_axis_values(axis_pools, random.Random(123))
    second = sample_axis_values(axis_pools, random.Random(123))

    assert first == second
    assert set(first.keys()) == {"style", "length", "context"}
    assert first["style"] in axis_pools["van_phong"]
    assert first["length"] in axis_pools["do_dai"]
    assert first["context"] in axis_pools["ngu_canh"]
