
"""Parser and validator for the educational ECQL v1.0 language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


FieldType = Literal["string", "number"]

ENTITY_FIELDS: dict[str, dict[str, FieldType]] = {
    "EMPLOYEES": {
        "name": "string",
        "city": "string",
        "department": "string",
        "position": "string",
        "status": "string",
        "salary": "number",
        "age": "number",
        "experience": "number",
    },
    "PROJECTS": {
        "name": "string",
        "status": "string",
        "department": "string",
        "priority": "string",
        "budget": "number",
        "team_size": "number",
        "progress": "number",
    },
    "INVENTORY": {
        "name": "string",
        "category": "string",
        "status": "string",
        "warehouse": "string",
        "price": "number",
        "quantity": "number",
        "stock": "number",
    },
    "DEALS": {
        "client": "string",
        "status": "string",
        "region": "string",
        "manager": "string",
        "amount": "number",
        "discount": "number",
        "probability": "number",
    },
}

COMPARISON_OPERATORS = {"IS", "NOT", "ABOVE", "BELOW"}
OUTPUT_FORMATS = {"JSON", "TABLE", "LIST"}

# Full query: FETCH [ENTITY] [WHERE ...] [AS FORMAT]
QUERY_PATTERN = re.compile(
    r"^FETCH \[([A-Z]+)\]"
    r"(?: WHERE (.+?))?"
    r"(?: AS (JSON|TABLE|LIST))?$"
)

# A value is either a single-quoted string without embedded quotes
# or a non-negative integer.
CONDITION_PATTERN = re.compile(
    r"^@([a-z][a-z0-9_]*) "
    r"(IS|NOT|ABOVE|BELOW) "
    r"('(?:[^']*)'|(?:0|[1-9][0-9]*))$"
)

LOGICAL_SEPARATOR_PATTERN = re.compile(r" (&&|\|\|) ")


class ECQLValidationError(ValueError):
    """Raised when an ECQL query violates the v1.0 specification."""


@dataclass(frozen=True)
class Condition:
    field: str
    operator: str
    value: str | int


@dataclass(frozen=True)
class ParsedQuery:
    entity: str
    conditions: tuple[Condition, ...]
    logical_operator: str | None
    output_format: str | None


def parse_condition(text: str, entity: str) -> Condition:
    """Parse and validate one comparison condition."""

    match = CONDITION_PATTERN.fullmatch(text)
    if match is None:
        raise ECQLValidationError(
            f"Invalid condition syntax: {text!r}"
        )

    field, operator, raw_value = match.groups()

    if entity not in ENTITY_FIELDS:
        raise ECQLValidationError(f"Unknown entity: {entity}")

    fields = ENTITY_FIELDS[entity]

    if field not in fields:
        raise ECQLValidationError(
            f"Field @{field} is not allowed for [{entity}]"
        )

    if operator not in COMPARISON_OPERATORS:
        raise ECQLValidationError(
            f"Unsupported comparison operator: {operator}"
        )

    if raw_value.startswith("'"):
        value: str | int = raw_value[1:-1]
        actual_type: FieldType = "string"
    else:
        value = int(raw_value)
        actual_type = "number"

    expected_type = fields[field]

    if actual_type != expected_type:
        raise ECQLValidationError(
            f"Field @{field} expects {expected_type}, "
            f"got {actual_type}"
        )

    if operator in {"ABOVE", "BELOW"} and expected_type != "number":
        raise ECQLValidationError(
            f"{operator} cannot be used with string field @{field}"
        )

    return Condition(
        field=field,
        operator=operator,
        value=value,
    )


def parse_ecql(query: str) -> ParsedQuery:
    """Parse a complete ECQL v1.0 query.

    Raises:
        ECQLValidationError: When the query is invalid.
    """

    if not isinstance(query, str):
        raise ECQLValidationError("Query must be a string")

    match = QUERY_PATTERN.fullmatch(query)

    if match is None:
        raise ECQLValidationError(
            "Query does not match ECQL v1.0 structure"
        )

    entity, where_clause, output_format = match.groups()

    if entity not in ENTITY_FIELDS:
        raise ECQLValidationError(f"Unknown entity: {entity}")

    if output_format is not None and output_format not in OUTPUT_FORMATS:
        raise ECQLValidationError(
            f"Unsupported output format: {output_format}"
        )

    if where_clause is None:
        return ParsedQuery(
            entity=entity,
            conditions=(),
            logical_operator=None,
            output_format=output_format,
        )

    parts = LOGICAL_SEPARATOR_PATTERN.split(where_clause)

    condition_texts = parts[::2]
    logical_operators = parts[1::2]

    if not condition_texts or any(not text for text in condition_texts):
        raise ECQLValidationError("Empty condition is not allowed")

    if len(set(logical_operators)) > 1:
        raise ECQLValidationError(
            "Mixing && and || is not supported in ECQL v1.0"
        )

    conditions = tuple(
        parse_condition(text, entity)
        for text in condition_texts
    )

    return ParsedQuery(
        entity=entity,
        conditions=conditions,
        logical_operator=(
            logical_operators[0] if logical_operators else None
        ),
        output_format=output_format,
    )


def validate_ecql(query: str) -> bool:
    """Return True for a valid ECQL query, otherwise False."""

    try:
        parse_ecql(query)
        return True
    except ECQLValidationError:
        return False
