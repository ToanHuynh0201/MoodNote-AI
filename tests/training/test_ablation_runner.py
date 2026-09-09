"""Tests cho phần so sánh/báo cáo ablation — hàm thuần, chạy được khi không có torch."""

import pytest

from src.training.ablation_runner import (
    aggregate_seeds,
    compare_scenarios,
    render_comparison_markdown,
)


def _result(scenario, accuracy, f1_macro, f1_weighted, n_train=100, smoke=False):
    return {
        "scenario": scenario,
        "smoke": smoke,
        "n_train": n_train,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        },
    }


def _results(combined=(0.70, 0.65, 0.69)):
    return {
        "real_only": _result("real_only", 0.66, 0.60, 0.65, n_train=5548),
        "synthetic_only": _result("synthetic_only", 0.50, 0.45, 0.49, n_train=5472),
        "combined": _result("combined", *combined, n_train=11020),
    }


def test_compare_scenarios_passes_when_combined_beats_baseline_on_every_metric():
    comparison = compare_scenarios(_results())

    assert comparison["passed"] is True
    assert comparison["failed_metrics"] == []
    assert comparison["synthetic_contribution"]["accuracy"] == pytest.approx(0.04)


def test_compare_scenarios_fails_when_combined_loses_on_one_metric():
    comparison = compare_scenarios(_results(combined=(0.70, 0.58, 0.69)))

    assert comparison["passed"] is False
    assert comparison["failed_metrics"] == ["f1_macro"]


def test_compare_scenarios_fails_on_a_tie_because_proposal_requires_strictly_higher():
    comparison = compare_scenarios(_results(combined=(0.66, 0.60, 0.65)))

    assert comparison["passed"] is False
    assert comparison["failed_metrics"] == ["accuracy", "f1_macro", "f1_weighted"]


def test_compare_scenarios_deltas_are_relative_to_the_named_baseline():
    comparison = compare_scenarios(_results(), baseline="synthetic_only")

    assert comparison["baseline"] == "synthetic_only"
    assert comparison["scenarios"]["synthetic_only"]["delta_vs_baseline"]["accuracy"] == 0.0
    assert comparison["scenarios"]["real_only"]["delta_vs_baseline"]["accuracy"] == pytest.approx(
        0.16
    )


def test_compare_scenarios_raises_when_baseline_result_is_missing():
    results = _results()
    del results["real_only"]

    with pytest.raises(ValueError, match="real_only"):
        compare_scenarios(results)


def test_compare_scenarios_marks_smoke_when_any_run_was_a_smoke_run():
    results = _results()
    results["combined"]["smoke"] = True

    assert compare_scenarios(results)["smoke"] is True


def test_render_comparison_markdown_reports_pass_verdict_and_all_scenarios():
    markdown = render_comparison_markdown(compare_scenarios(_results()))

    assert "**ĐẠT**" in markdown
    for scenario in ("real_only", "synthetic_only", "combined"):
        assert f"`{scenario}`" in markdown
    assert "11,020" in markdown
    assert "+0.0400" in markdown
    assert "Giới hạn phương pháp" in markdown


def test_render_comparison_markdown_reports_fail_verdict_with_failing_metric():
    markdown = render_comparison_markdown(compare_scenarios(_results(combined=(0.70, 0.58, 0.69))))

    assert "**CHƯA ĐẠT**" in markdown
    assert "f1_macro" in markdown


def test_render_comparison_markdown_warns_loudly_on_smoke_results():
    results = _results()
    results["combined"]["smoke"] = True

    markdown = render_comparison_markdown(compare_scenarios(results))

    assert "CHẠY SMOKE" in markdown


def _seed_run(scenario, accuracy, f1_macro, f1_weighted, seed, n_train=100):
    run = _result(scenario, accuracy, f1_macro, f1_weighted, n_train=n_train)
    run["seed"] = seed
    run["model_name"] = "vinai/phobert-base-v2"
    return run


def test_aggregate_seeds_averages_overall_metrics():
    runs = [
        _seed_run("real_only", 0.60, 0.55, 0.59, seed=42, n_train=5548),
        _seed_run("real_only", 0.62, 0.57, 0.61, seed=43, n_train=5548),
        _seed_run("real_only", 0.64, 0.59, 0.63, seed=44, n_train=5548),
    ]

    agg = aggregate_seeds(runs)

    assert agg["scenario"] == "real_only"
    assert agg["n_seeds"] == 3
    assert agg["seeds"] == [42, 43, 44]
    assert agg["n_train"] == 5548
    assert agg["metrics"]["accuracy"] == pytest.approx(0.62)
    assert agg["metrics"]["f1_macro"] == pytest.approx(0.57)


def test_aggregate_seeds_reports_population_std():
    import statistics

    accs = [0.60, 0.62, 0.64]
    runs = [
        _seed_run("combined", a, 0.5, 0.5, seed=s) for a, s in zip(accs, (1, 2, 3), strict=True)
    ]

    agg = aggregate_seeds(runs)

    assert agg["metrics_std"]["accuracy"] == pytest.approx(statistics.pstdev(accs))
    assert agg["metrics_std"]["f1_macro"] == pytest.approx(0.0)


def test_aggregate_seeds_single_run_has_zero_std():
    agg = aggregate_seeds([_seed_run("synthetic_only", 0.30, 0.28, 0.31, seed=42)])

    assert agg["n_seeds"] == 1
    assert agg["metrics"]["accuracy"] == pytest.approx(0.30)
    assert all(v == pytest.approx(0.0) for v in agg["metrics_std"].values())


def test_aggregate_seeds_rejects_empty_list():
    with pytest.raises(ValueError, match="rỗng"):
        aggregate_seeds([])


def test_render_comparison_markdown_shows_mean_and_std_when_multi_seed():
    comparison = compare_scenarios(_results())
    comparison["n_seeds"] = {"real_only": 3, "synthetic_only": 1, "combined": 3}
    comparison["scenarios"]["combined"]["scores_std"] = {
        "accuracy": 0.008,
        "f1_macro": 0.010,
        "f1_weighted": 0.009,
    }
    comparison["mcnemar"] = {
        "seed": 42,
        "n": 693,
        "only_base_correct": 20,
        "only_treat_correct": 45,
        "p_value": 0.0032,
    }

    markdown = render_comparison_markdown(comparison)

    assert "0.7000 ± 0.0080" in markdown
    assert "3 seed" in markdown
    assert "McNemar" in markdown
    assert "0.0032" in markdown
