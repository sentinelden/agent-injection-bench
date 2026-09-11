"""Corpus validation.

A corpus that silently drops malformed lines produces scores that look better
than reality, so every one of these asserts a hard failure.
"""

import json
import pytest

from aib.dataset import DatasetError, load


def write(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


VALID = {
    "id": "a-1", "channel": "tool_result", "family": "f", "task": "t",
    "injected": "i", "success": {"kind": "canary", "canary": "C1"},
}


def test_loads_valid_corpus(tmp_path):
    attacks = load(write(tmp_path, [VALID]))
    assert len(attacks) == 1
    assert attacks[0].id == "a-1"


def test_bundled_corpus_is_valid():
    from aib.cli import DEFAULT_CORPUS
    attacks = load(DEFAULT_CORPUS)
    assert len(attacks) >= 20
    # Controls are what make the over-refusal number possible.
    assert any(a.is_control for a in attacks)


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(DatasetError, match="duplicate"):
        load(write(tmp_path, [VALID, VALID]))


def test_rejects_unknown_channel(tmp_path):
    bad = {**VALID, "channel": "telepathy"}
    with pytest.raises(DatasetError, match="unknown channel"):
        load(write(tmp_path, [bad]))


def test_rejects_missing_field(tmp_path):
    bad = {k: v for k, v in VALID.items() if k != "task"}
    with pytest.raises(DatasetError, match="missing required field"):
        load(write(tmp_path, [bad]))


def test_rejects_criterion_without_its_fields(tmp_path):
    bad = {**VALID, "success": {"kind": "exfiltration", "sink": "http_post"}}
    with pytest.raises(DatasetError, match="requires"):
        load(write(tmp_path, [bad]))


def test_rejects_tool_not_in_registry(tmp_path):
    """An attack naming an absent tool can never succeed.

    Left unchecked this looks identical to a defense working, which is the
    most dangerous kind of corpus bug.
    """
    bad = {**VALID, "success": {"kind": "tool_call", "tool": "delete_file"}, "tools": []}
    with pytest.raises(DatasetError, match="not in this attack's tool registry"):
        load(write(tmp_path, [bad]))


def test_rejects_empty_corpus(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load(path)
