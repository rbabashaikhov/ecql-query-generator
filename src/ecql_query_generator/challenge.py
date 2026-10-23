"""Manually authored natural-language holdout; never use for training."""
from __future__ import annotations

import json
from pathlib import Path

from ecql_query_generator.generator import INSTRUCTION
from ecql_query_generator.validator import parse_ecql

# Labels are deliberately closer to how a person might speak than training templates.
CASES = [
    ("Кто получает больше 150 тысяч?", "FETCH [EMPLOYEES] WHERE @salary ABOVE 150000"),
    ("Выведи работников младше 30 лет", "FETCH [EMPLOYEES] WHERE @age BELOW 30"),
    ("Сотрудники из Казани, только не из отдела HR", "FETCH [EMPLOYEES] WHERE @city IS 'Kazan' && @department NOT 'hr'"),
    ("Хочу всех работников, у кого зарплата ниже 80 тысяч или стаж выше пяти лет", "FETCH [EMPLOYEES] WHERE @salary BELOW 80000 || @experience ABOVE 5"),
    ("Дай полный перечень сотрудников в JSON", "FETCH [EMPLOYEES] AS JSON"),
    ("Какие проекты сейчас активны?", "FETCH [PROJECTS] WHERE @status IS 'active'"),
    ("Проекты с бюджетом более миллиона", "FETCH [PROJECTS] WHERE @budget ABOVE 1000000"),
    ("Покажи проекты не с высоким приоритетом", "FETCH [PROJECTS] WHERE @priority NOT 'high'"),
    ("Нужны проекты со средней приоритетностью и командой больше 10 человек", "FETCH [PROJECTS] WHERE @priority IS 'medium' && @team_size ABOVE 10"),
    ("Все проекты табличкой, без фильтров", "FETCH [PROJECTS] AS TABLE"),
    ("Товары дешевле 5000 рублей", "FETCH [INVENTORY] WHERE @price BELOW 5000"),
    ("Покажи зарезервированные товары списком", "FETCH [INVENTORY] WHERE @status IS 'reserved' AS LIST"),
    ("Товары с остатком больше 10 или с количеством меньше пяти", "FETCH [INVENTORY] WHERE @stock ABOVE 10 || @quantity BELOW 5"),
    ("Неэлектронные товары с ценой ровно 10000", "FETCH [INVENTORY] WHERE @category NOT 'electronics' && @price IS 10000"),
    ("Выгрузи весь склад в JSON", "FETCH [INVENTORY] AS JSON"),
    ("Сделки на сумму свыше 500 тысяч", "FETCH [DEALS] WHERE @amount ABOVE 500000"),
    ("Покажи проигранные сделки", "FETCH [DEALS] WHERE @status IS 'lost'"),
    ("Сделки не из Сибири и с вероятностью ниже 25", "FETCH [DEALS] WHERE @region NOT 'Siberia' && @probability BELOW 25"),
    ("Сделки менеджера Анны или сделки на сумму ровно миллион", "FETCH [DEALS] WHERE @manager IS 'Anna' || @amount IS 1000000"),
    ("Полный список сделок без условий", "FETCH [DEALS] AS LIST"),
]


def generate_challenge(output_dir: str | Path = "data") -> int:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    assert len(CASES) == 20
    assert len({r[0] for r in CASES}) == len(CASES)
    assert all(parse_ecql(ecql) for _, ecql in CASES)
    with (path / "challenge.jsonl").open("w", encoding="utf-8") as fh:
        for ru, ecql in CASES:
            fh.write(json.dumps({"instruction": INSTRUCTION, "input": ru, "output": ecql}, ensure_ascii=False) + "\n")
    return len(CASES)


if __name__ == "__main__":
    print(f"Created {generate_challenge()} challenge records")
