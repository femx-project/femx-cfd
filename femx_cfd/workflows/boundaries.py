"""Translate portable fixed boundaries into model problem conditions."""

from femx_cfd._typing import Config


def fixed_conditions(case: Config, dimension: int) -> list[Config]:
    """Return explicit uniform conditions for a model problem."""

    conditions = []
    for condition in case["boundary_conditions"]:
        kind = condition["kind"]
        value = condition["value"]
        if kind == "velocity" and len(value) != dimension:
            boundary = condition["boundary"]
            raise ValueError(
                f"velocity boundary '{boundary}' must contain {dimension} values"
            )
        conditions.append(
            {
                "boundary": condition["boundary"],
                "field": kind,
                "space": {"kind": "uniform", "value": value},
                "time": {"kind": "uniform"},
            }
        )
    return conditions


__all__ = ["fixed_conditions"]
