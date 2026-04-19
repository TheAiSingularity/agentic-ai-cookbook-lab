"""Unit tests for ablation.py and pareto.py — pure data-path tests, no subprocess."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ablation import CONFIGS, _load_dataset, _load_existing
from pareto import aggregate


def test_configs_have_required_labels():
    """Every default config has a unique label, description, and main script."""
    labels = {c.label for c in CONFIGS.values()}
    assert labels == {"A1", "A2", "A3", "B1", "B2", "B3", "B4"}
    for label, cfg in CONFIGS.items():
        assert cfg.label == label
        assert cfg.description
        assert cfg.main.suffix == ".py"


def test_load_existing_handles_missing_file(tmp_path):
    assert _load_existing(tmp_path / "nope.jsonl") == set()


def test_load_existing_parses_keys(tmp_path):
    f = tmp_path / "results.jsonl"
    f.write_text(
        json.dumps({"question_id": "q1", "config": "A1", "seed": 0}) + "\n" +
        json.dumps({"question_id": "q1", "config": "A2", "seed": 0}) + "\n"
    )
    done = _load_existing(f)
    assert ("q1", "A1", 0) in done
    assert ("q1", "A2", 0) in done
    assert ("q2", "A1", 0) not in done


def test_load_existing_skips_malformed_lines(tmp_path):
    f = tmp_path / "results.jsonl"
    f.write_text(
        json.dumps({"question_id": "q1", "config": "A1", "seed": 0}) + "\n" +
        "not json\n" +
        json.dumps({"config": "A1"}) + "\n"  # missing question_id
    )
    assert _load_existing(f) == {("q1", "A1", 0)}


def test_load_dataset_parses_jsonl(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text(
        json.dumps({"id": "x", "question": "Q?", "gold_answer": "A.", "source": "test"}) + "\n" +
        json.dumps({"id": "y", "question": "Q2?", "gold_answer": "B.", "source": "test"}) + "\n"
    )
    ds = _load_dataset(f)
    assert len(ds) == 2
    assert ds[0]["id"] == "x"


def test_aggregate_computes_mean_per_config():
    rows = [
        {"source": "SimpleQA", "config": "A1", "factuality": 1.0, "citation_precision": 1.0,
         "citation_accuracy": 1.0, "latency_s": 40.0, "tokens_est": 500},
        {"source": "SimpleQA", "config": "A1", "factuality": 0.5, "citation_precision": 0.5,
         "citation_accuracy": 1.0, "latency_s": 60.0, "tokens_est": 700},
        {"source": "SimpleQA", "config": "B3", "factuality": 1.0, "citation_precision": 1.0,
         "citation_accuracy": 1.0, "latency_s": 90.0, "tokens_est": 1200},
    ]
    agg = aggregate(rows)
    a1 = agg[("SimpleQA", "A1")]
    b3 = agg[("SimpleQA", "B3")]
    assert a1["n"] == 2
    assert a1["factuality_mean"] == 0.75
    assert b3["factuality_mean"] == 1.0


def test_aggregate_handles_empty_source():
    rows = [{"config": "A1", "factuality": 1.0, "citation_precision": 1.0,
             "citation_accuracy": 1.0, "latency_s": 10.0, "tokens_est": 100}]
    agg = aggregate(rows)
    assert ("", "A1") in agg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
