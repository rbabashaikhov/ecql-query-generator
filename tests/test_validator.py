
"""Tests for the ECQL v1.0 validator."""

import pytest

from ecql_query_generator.validator import (
    ECQLValidationError,
    parse_ecql,
    validate_ecql,
)


VALID_QUERIES = [
    "FETCH [EMPLOYEES]",
    "FETCH [EMPLOYEES] AS JSON",
    "FETCH [PROJECTS] WHERE @status IS 'active'",
    "FETCH [EMPLOYEES] WHERE @salary ABOVE 150000",
    "FETCH [EMPLOYEES] WHERE @city IS 'Moscow' && @salary ABOVE 150000",
    "FETCH [PROJECTS] WHERE @status IS 'active' || @status IS 'planned'",
    "FETCH [INVENTORY] WHERE @price BELOW 50000 && @quantity ABOVE 10 AS TABLE",
    "FETCH [DEALS] WHERE @amount ABOVE 1000000 AS JSON",
    "FETCH [EMPLOYEES] WHERE @salary NOT 100000 AS LIST",
    "FETCH [PROJECTS] WHERE @budget IS 0",
]

INVALID_QUERIES = [
    "SELECT * FROM employees",
    "FETCH EMPLOYEES",
    "FETCH [CUSTOMERS]",
    "FETCH [EMPLOYEES] WHERE salary ABOVE 100000",
    "FETCH [EMPLOYEES] WHERE @salary > 100000",
    "FETCH [EMPLOYEES] WHERE @salary >= 100000",
    "FETCH [EMPLOYEES] WHERE @salary ABOVE '100000'",
    "FETCH [EMPLOYEES] WHERE @city ABOVE 'Moscow'",
    "FETCH [EMPLOYEES] WHERE @salary IS 'high'",
    "FETCH [PROJECTS] WHERE @salary ABOVE 100000",
    "FETCH [EMPLOYEES] WHERE @city IS Moscow",
    "FETCH [EMPLOYEES] WHERE @salary ABOVE -100",
    "FETCH [EMPLOYEES] WHERE @salary ABOVE 10.5",
    "FETCH [EMPLOYEES] WHERE @salary ABOVE 100 && @age BELOW 50 || @city IS 'Moscow'",
    "FETCH [EMPLOYEES] WHERE (@salary ABOVE 100)",
    "FETCH [EMPLOYEES] ORDER BY @salary",
    "FETCH [DEALS] WHERE @amount ABOVE 100 AS CSV",
    "FETCH [EMPLOYEES] WHERE @salary ABOVE 100 &&",
    "FETCH [EMPLOYEES] WHERE",
    "",
]


@pytest.mark.parametrize("query", VALID_QUERIES)
def test_valid_queries(query: str) -> None:
    assert validate_ecql(query) is True


@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_invalid_queries(query: str) -> None:
    assert validate_ecql(query) is False


def test_parsed_query_structure() -> None:
    query = (
        "FETCH [EMPLOYEES] "
        "WHERE @city IS 'Moscow' && @salary ABOVE 150000 "
        "AS TABLE"
    )

    result = parse_ecql(query)

    assert result.entity == "EMPLOYEES"
    assert result.output_format == "TABLE"
    assert result.logical_operator == "&&"
    assert len(result.conditions) == 2

    assert result.conditions[0].field == "city"
    assert result.conditions[0].operator == "IS"
    assert result.conditions[0].value == "Moscow"

    assert result.conditions[1].field == "salary"
    assert result.conditions[1].operator == "ABOVE"
    assert result.conditions[1].value == 150000


def test_invalid_query_raises_exception() -> None:
    with pytest.raises(ECQLValidationError):
        parse_ecql("SELECT * FROM employees")


def test_simple_query_has_no_conditions() -> None:
    result = parse_ecql("FETCH [PROJECTS]")

    assert result.conditions == ()
    assert result.logical_operator is None
    assert result.output_format is None
