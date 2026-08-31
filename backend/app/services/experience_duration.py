"""Deterministic calculation of total professional experience from dated entries.

Does not depend on the LLM to emit `total_years_experience`. Parses start/end
pairs from structured `ExperienceItem` fields, merges overlapping intervals,
and returns the unique months of employment converted to years.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from app.models.schemas import CandidateProfile, ExperienceItem

# YYYY-MM, YYYY.MM, YYYY/MM (with optional Chinese year/month markers).
_DATE_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})\s*[-./年]?\s*(?P<month>1[0-2]|0?[1-9])\s*月?"
)

_PRESENT_RE = re.compile(
    r"至今|现在|目前|Present|Current|Now",
    re.IGNORECASE,
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _parse_date(text: str) -> tuple[int, int] | None:
    """Parse the first YYYY-MM / YYYY.MM / YYYY/MM date in text."""
    if not text:
        return None
    match = _DATE_RE.search(text)
    if not match:
        return None
    return int(match.group("year")), int(match.group("month"))


def _is_present(text: str | None) -> bool:
    if not text:
        return False
    return _PRESENT_RE.search(text) is not None


@dataclass(frozen=True)
class ExperienceInterval:
    start_month: int
    end_month: int
    raw_start: str
    raw_end: str


@dataclass(frozen=True)
class ExperienceDuration:
    total_months: int
    total_years: int
    intervals: tuple[ExperienceInterval, ...]
    merged_intervals: tuple[tuple[int, int], ...]
    overlap_months_removed: int
    source: str


class ExperienceDurationService:
    """Calculate total experience duration deterministically from resume data."""

    def __init__(self, today: date | None = None):
        self._today = today or date.today()

    def compute(self, profile: CandidateProfile) -> ExperienceDuration | None:
        """Return total experience duration, or None if no parseable dates."""
        intervals: list[ExperienceInterval] = []
        for item in profile.experience:
            interval = self._interval_from_item(item)
            if interval:
                intervals.append(interval)

        if not intervals:
            return None

        merged, overlap_removed = self._merge_intervals(
            [(i.start_month, i.end_month) for i in intervals]
        )
        total_months = sum(end - start + 1 for start, end in merged)
        return ExperienceDuration(
            total_months=total_months,
            total_years=max(0, total_months // 12),
            intervals=tuple(intervals),
            merged_intervals=tuple(merged),
            overlap_months_removed=overlap_removed,
            source="deterministic_dates",
        )

    def total_years(self, profile: CandidateProfile) -> int | None:
        """Return total years of experience, or None if dates cannot be parsed."""
        duration = self.compute(profile)
        if duration is None:
            return None
        return duration.total_years

    def _interval_from_item(
        self, item: ExperienceItem
    ) -> ExperienceInterval | None:
        start = _parse_date(item.start_date or "")
        if start is None:
            return None
        start_year, start_month = start
        start_idx = _month_index(start_year, start_month)

        if _is_present(item.end_date):
            end_idx = _month_index(self._today.year, self._today.month)
        else:
            end = _parse_date(item.end_date or "")
            if end is None:
                return None
            end_year, end_month = end
            end_idx = _month_index(end_year, end_month)

        if end_idx < start_idx:
            return None

        return ExperienceInterval(
            start_month=start_idx,
            end_month=end_idx,
            raw_start=item.start_date or "",
            raw_end=item.end_date or "",
        )

    @staticmethod
    def _merge_intervals(
        intervals: Iterable[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], int]:
        """Merge overlapping intervals and report how many overlap months were removed."""
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        if not sorted_intervals:
            return [], 0

        merged: list[tuple[int, int]] = []
        overlap_removed = 0
        current_start, current_end = sorted_intervals[0]

        for start, end in sorted_intervals[1:]:
            if start <= current_end + 1:
                # Overlapping or adjacent — extend the current interval.
                if end > current_end:
                    overlap_removed += current_end - start + 1
                    current_end = end
                else:
                    # Fully contained within current interval.
                    overlap_removed += end - start + 1
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        return merged, overlap_removed
