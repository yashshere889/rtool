"""How numbers are printed everywhere (tables, placeholders, figures), so the same value always reads the same."""

from __future__ import annotations

import math


def fmt(value: float | None, digits: int = 3) -> str:
    """``digits`` significant figures, no scientific notation for everyday magnitudes."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    if magnitude >= 6 or magnitude <= -5:
        return f"{value:.{digits - 1}e}"
    decimals = max(0, digits - 1 - magnitude)
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def fmt_ci(low: float | None, high: float | None, digits: int = 3) -> str:
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"
