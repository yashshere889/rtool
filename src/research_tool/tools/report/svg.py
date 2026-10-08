"""Tiny SVG bar charts with error bars: deterministic figures without matplotlib."""

from __future__ import annotations

import math
from html import escape

from .numbers import fmt

WIDTH, HEIGHT = 640, 380
LEFT, RIGHT, TOP, BOTTOM = 72, 24, 48, 96


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        high = low + (abs(low) or 1.0)
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if raw <= m * magnitude)
    ticks = [round(math.floor(low / step) * step, 12)]
    while ticks[-1] < high - step * 1e-9:  # the axis must reach past the highest value
        ticks.append(round(ticks[-1] + step, 12))
    return ticks


def bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    lows: list[float],
    highs: list[float],
    *,
    ylabel: str = "",
    highlight: int | None = None,
    note: str = "",
) -> str:
    """Bars with error bars (e.g. means and 95 % confidence intervals), one per label."""
    ticks = nice_ticks(min(0.0, *lows, *values), max(0.0, *highs, *values))
    y_min, y_max = ticks[0], ticks[-1]
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def y(v: float) -> float:
        return TOP + plot_h * (1 - (v - y_min) / (y_max - y_min))

    slot = plot_w / max(1, len(labels))
    bar_w = slot * 0.6
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="{WIDTH / 2}" y="24" text-anchor="middle" font-size="15" font-weight="bold">{escape(title)}</text>',
    ]
    for tick in ticks:
        ty = y(tick)
        parts.append(f'<line x1="{LEFT}" y1="{ty:.1f}" x2="{WIDTH - RIGHT}" y2="{ty:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{LEFT - 8}" y="{ty + 4:.1f}" text-anchor="end" fill="#444">{fmt(tick)}</text>')
    parts.append(f'<line x1="{LEFT}" y1="{y(0):.1f}" x2="{WIDTH - RIGHT}" y2="{y(0):.1f}" stroke="#888"/>')
    for i, (label, value, low, high) in enumerate(zip(labels, values, lows, highs, strict=True)):
        cx = LEFT + slot * (i + 0.5)
        top, base = y(max(value, 0)), y(min(value, 0))
        colour = "#9e9e9e" if i == highlight else "#3b6ea8"
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{max(base - top, 0.5):.1f}" fill="{colour}"/>'
        )
        y_low, y_high = y(low), y(high)
        parts.append(f'<line x1="{cx:.1f}" y1="{y_low:.1f}" x2="{cx:.1f}" y2="{y_high:.1f}" stroke="#222"/>')
        for cap in (y_low, y_high):
            parts.append(f'<line x1="{cx - 6:.1f}" y1="{cap:.1f}" x2="{cx + 6:.1f}" y2="{cap:.1f}" stroke="#222"/>')
        short = label if len(label) <= 18 else label[:17] + "…"
        parts.append(
            f'<text x="{cx:.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="end" '
            f'transform="rotate(-30 {cx:.1f} {HEIGHT - BOTTOM + 18})">{escape(short)}</text>'
        )
    if ylabel:
        parts.append(
            f'<text x="16" y="{TOP + plot_h / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 16 {TOP + plot_h / 2:.1f})">{escape(ylabel)}</text>'
        )
    if note:
        parts.append(f'<text x="{LEFT}" y="{HEIGHT - 10}" fill="#666" font-size="11">{escape(note)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
