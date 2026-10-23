"""Deterministic ECQL v2 corpus: structured gold first, Russian surface second."""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ecql_query_generator.validator import ENTITY_FIELDS, parse_ecql

SEED = 42
INSTRUCTION = "Переведи запрос на ECQL"
FORMATS = [None, "JSON", "TABLE", "LIST"]
FORMAT_RU = {None: "", "JSON": "в формате JSON", "TABLE": "в виде таблицы", "LIST": "списком"}
SUBJECTS = {"EMPLOYEES": "сотрудников", "PROJECTS": "проекты", "INVENTORY": "товары", "DEALS": "сделки"}

# These Russian glosses are part of our synthetic ECQL teaching schema, not the assignment.
TEXT_VALUES = {
    "EMPLOYEES": {
        "city": [("Moscow", "Москва"), ("Kazan", "Казань"), ("Sochi", "Сочи"), ("Perm", "Пермь")],
        "department": [("sales", "продажи"), ("it", "IT"), ("finance", "финансы"), ("hr", "HR")],
        "status": [("active", "активный"), ("inactive", "неактивный")],
    },
    "PROJECTS": {
        "status": [("active", "активный"), ("planned", "запланированный"), ("completed", "завершённый")],
        "priority": [("high", "высокий"), ("medium", "средний"), ("low", "низкий")],
        "department": [("sales", "продажи"), ("it", "IT"), ("finance", "финансы")],
    },
    "INVENTORY": {
        "category": [("electronics", "электроника"), ("furniture", "мебель"), ("clothing", "одежда")],
        "status": [("available", "доступен"), ("reserved", "зарезервирован"), ("sold", "продан")],
        "warehouse": [("central", "центральный"), ("north", "северный")],
    },
    "DEALS": {
        "region": [("Moscow", "Москва"), ("Siberia", "Сибирь"), ("Ural", "Урал")],
        "status": [("open", "открыта"), ("won", "выиграна"), ("lost", "проиграна")],
        "manager": [("Anna", "Анна"), ("Ivan", "Иван")],
    },
}
NUMBERS = {
    "salary": [50000, 80000, 100000, 150000, 200000],
    "age": [21, 25, 30, 40, 50],
    "experience": [1, 3, 5, 8, 12],
    "budget": [100000, 300000, 500000, 1000000],
    "team_size": [3, 5, 10, 20],
    "progress": [10, 30, 50, 90],
    "price": [1000, 5000, 10000, 50000],
    "quantity": [5, 10, 20, 100],
    "stock": [5, 10, 50, 100],
    "amount": [100000, 500000, 1000000, 3000000],
    "discount": [5, 10, 15, 25],
    "probability": [10, 25, 50, 75],
}
FEMININE = {"salary", "price", "quantity", "amount", "discount", "probability", "category"}
LABELS = {
    "city": "город", "department": "отдел", "status": "статус",
    "priority": "приоритет", "region": "регион", "manager": "менеджер",
    "category": "категория", "warehouse": "склад", "salary": "зарплата",
    "age": "возраст", "experience": "стаж", "budget": "бюджет",
    "team_size": "размер команды", "progress": "прогресс", "price": "цена",
    "quantity": "количество", "stock": "остаток", "amount": "сумма",
    "discount": "скидка", "probability": "вероятность",
}

@dataclass(frozen=True)
class ConditionSpec:
    field: str
    operator: str
    value: str | int
    russian: str

    def ecql(self) -> str:
        literal = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
        return f"@{self.field} {self.operator} {literal}"


def random_condition(entity: str, field: str, rng: random.Random) -> ConditionSpec:
    label = LABELS[field]
    if ENTITY_FIELDS[entity][field] == "string":
        value, ru = rng.choice(TEXT_VALUES[entity][field])
        op = rng.choice(["IS", "NOT"])
        phrase = (f"{label} — «{ru}»" if op == "IS"
                  else f"{label} не {'равна' if field in FEMININE else 'равен'} значению «{ru}»")
    else:
        value = rng.choice(NUMBERS[field])
        op = rng.choice(["IS", "NOT", "ABOVE", "BELOW"])
        phrase = {
            "IS": f"{label} {'равна' if field in FEMININE else 'равен'} {value}",
            "NOT": f"{label} не {'равна' if field in FEMININE else 'равен'} {value}",
            "ABOVE": f"{label} строго больше {value}",
            "BELOW": f"{label} строго меньше {value}",
        }[op]
    return ConditionSpec(field, op, value, phrase)


def render_ru(entity: str, clauses: list[str], connector: str | None,
              fmt: str | None, family: int) -> str:
    subject = SUBJECTS[entity]
    if not clauses:
        leads = [f"Найди {subject} без фильтров", f"Покажи {subject} без условий",
                 f"Выведи {subject} целиком", f"Покажи {subject} без отбора",
                 f"Отобрази {subject} без фильтрации", f"Дай полный перечень: {subject}"]
        request = leads[family]
    else:
        separator = " и одновременно " if connector == "&&" else " или "
        criteria = separator.join(clauses)
        leads = [f"Найди {subject}, у которых {criteria}",
                 f"Покажи {subject}: {criteria}",
                 f"Покажи {subject}, где {criteria}",
                 f"Выведи {subject} по условиям: {criteria}",
                 f"Отобрази {subject}; критерии: {criteria}",
                 f"Подбери {subject} по критериям: {criteria}"]
        request = leads[family]
    if fmt:
        request += f"; результат {FORMAT_RU[fmt]}"
    return request


def make_example(entity: str, family: int, rng: random.Random, number: int) -> dict:
    fields = list(TEXT_VALUES[entity]) + [
        field for field in NUMBERS if field in ENTITY_FIELDS[entity]
    ]
    condition_count = 0 if number % 25 == 0 else 1 + (number % 3)
    chosen = rng.sample(fields, condition_count)
    conditions = [random_condition(entity, field, rng) for field in chosen]
    connector = rng.choice(["&&", "||"]) if condition_count > 1 else None
    fmt = FORMATS[(number + family) % len(FORMATS)]
    ecql = f"FETCH [{entity}]"
    if conditions:
        ecql += " WHERE " + (f" {connector} " if connector else "").join(
            condition.ecql() for condition in conditions
        )
    if fmt:
        ecql += f" AS {fmt}"
    ru = render_ru(entity, [c.russian for c in conditions], connector, fmt, family)
    parsed = parse_ecql(ecql)
    assert len(parsed.conditions) == len(conditions)
    assert all(a.field == b.field and a.operator == b.operator and a.value == b.value
               for a, b in zip(parsed.conditions, conditions))
    return {"instruction": INSTRUCTION, "input": ru, "output": ecql}


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_dataset(output_dir: str | Path = "data") -> dict:
    rng = random.Random(SEED)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test = [], []
    seen_ru, seen_ecql = set(), set()
    entity_breakdown = {}
    for entity in SUBJECTS:
        selected = {"train": 0, "test": 0}
        attempts = 0
        while selected["train"] < 40 or selected["test"] < 10:
            if attempts >= 100_000:
                raise RuntimeError(f"Could not generate enough unique examples for {entity}")
            # Two wording families are held out of training entirely.
            target = "train" if selected["train"] < 40 else "test"
            family = rng.choice(range(4)) if target == "train" else rng.choice([4, 5])
            example = make_example(entity, family, rng, attempts)
            attempts += 1
            if example["input"] in seen_ru or example["output"] in seen_ecql:
                continue
            seen_ru.add(example["input"])
            seen_ecql.add(example["output"])
            (train if target == "train" else test).append(example)
            selected[target] += 1
        entity_breakdown[entity] = selected
    rng.shuffle(train)
    rng.shuffle(test)
    full = train + test
    assert (len(train), len(test), len(full)) == (160, 40, 200)
    for name, items in (("train", train), ("test", test), ("dataset_full", full)):
        _write_jsonl(output / f"{name}.jsonl", items)
    stats = {
        "seed": SEED, "total": len(full), "train": len(train), "test": len(test),
        "by_entity": entity_breakdown,
        "formats": dict(Counter(
            next((fmt for fmt in FORMATS[1:] if item["output"].endswith(f"AS {fmt}")), "DEFAULT")
            for item in full)),
        "valid_ecql": all(parse_ecql(item["output"]) for item in full),
        "held_out_surface_families": [4, 5],
        "caveat": "Surface-family holdout is not proof of semantic generalization; audit manually.",
    }
    (output / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


if __name__ == "__main__":
    print(json.dumps(generate_dataset(), ensure_ascii=False, indent=2))
