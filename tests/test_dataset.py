"""Quality gates for reproducibility, split integrity, and ECQL syntax."""
import json
from pathlib import Path

from ecql_query_generator.generator import generate_dataset
from ecql_query_generator.validator import parse_ecql


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sizes_syntax_and_distinctness(tmp_path):
    stats = generate_dataset(tmp_path)
    train = read_jsonl(tmp_path / "train.jsonl")
    test = read_jsonl(tmp_path / "test.jsonl")
    full = train + test
    assert stats["total"] == 200
    assert (len(train), len(test)) == (160, 40)
    assert len({r["input"] for r in full}) == 200
    assert len({r["output"] for r in full}) == 200
    assert {r["input"] for r in train}.isdisjoint({r["input"] for r in test})
    assert {r["output"] for r in train}.isdisjoint({r["output"] for r in test})
    assert all(r["instruction"] == "Переведи запрос на ECQL" for r in full)
    assert all(parse_ecql(r["output"]) for r in full)
    for entity in ("EMPLOYEES", "PROJECTS", "INVENTORY", "DEALS"):
        assert sum(f"[{entity}]" in r["output"] for r in train) == 40
        assert sum(f"[{entity}]" in r["output"] for r in test) == 10


def test_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(a)
    generate_dataset(b)
    for filename in ("train.jsonl", "test.jsonl", "dataset_full.jsonl", "dataset_stats.json"):
        assert (a / filename).read_bytes() == (b / filename).read_bytes()


def test_held_out_surface_leads(tmp_path):
    generate_dataset(tmp_path)
    train = read_jsonl(tmp_path / "train.jsonl")
    test = read_jsonl(tmp_path / "test.jsonl")
    assert all(not r["input"].startswith(("Подбери ", "Отобрази ", "Дай полный перечень:")) for r in train)
    assert all(r["input"].startswith(("Подбери ", "Отобрази ", "Дай полный перечень:")) for r in test)


def test_challenge_valid_and_separate(tmp_path):
    from ecql_query_generator.challenge import generate_challenge

    generate_dataset(tmp_path)
    assert generate_challenge(tmp_path) == 20
    main = read_jsonl(tmp_path / "dataset_full.jsonl")
    challenge = read_jsonl(tmp_path / "challenge.jsonl")
    assert len(challenge) == 20
    assert all(parse_ecql(row["output"]) for row in challenge)
    assert {r["input"] for r in main}.isdisjoint({r["input"] for r in challenge})
