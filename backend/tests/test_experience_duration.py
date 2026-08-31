"""Tests for deterministic experience duration calculation."""

from datetime import date

from app.models.schemas import CandidateProfile, ExperienceItem
from app.services.experience_duration import ExperienceDurationService


def test_two_sequential_roles_sum_to_total_months():
    profile = CandidateProfile(
        experience=[
            ExperienceItem(
                title="Role A",
                company="Corp A",
                start_date="2022.06",
                end_date="2022.12",
            ),
            ExperienceItem(
                title="Role B",
                company="Corp B",
                start_date="2023.06",
                end_date="2023.12",
            ),
        ]
    )
    service = ExperienceDurationService(today=date(2024, 1, 1))
    duration = service.compute(profile)
    assert duration is not None
    # Each role spans 7 inclusive months (Jun-Dec). They are sequential, so
    # the total unique months are 7 + 7 = 14.
    assert duration.total_months == 14
    assert duration.total_years == 1


def test_overlapping_roles_do_not_double_count():
    profile = CandidateProfile(
        experience=[
            ExperienceItem(
                title="Role A",
                company="Corp A",
                start_date="2022.01",
                end_date="2022.12",
            ),
            ExperienceItem(
                title="Role B",
                company="Corp B",
                start_date="2022.06",
                end_date="2023.06",
            ),
        ]
    )
    service = ExperienceDurationService(today=date(2024, 1, 1))
    duration = service.compute(profile)
    assert duration is not None
    # 2022-01 to 2023-06 inclusive = 18 months
    assert duration.total_months == 18
    assert duration.total_years == 1


def test_current_role_uses_today():
    profile = CandidateProfile(
        experience=[
            ExperienceItem(
                title="Current Role",
                company="Corp",
                start_date="2023.01",
                end_date="至今",
            ),
        ]
    )
    service = ExperienceDurationService(today=date(2024, 6, 1))
    duration = service.compute(profile)
    assert duration is not None
    # 2023-01 to 2024-06 inclusive = 18 months
    assert duration.total_months == 18
    assert duration.total_years == 1


def test_unparseable_dates_return_none():
    profile = CandidateProfile(
        experience=[
            ExperienceItem(
                title="Role",
                company="Corp",
                start_date="present",
                end_date="now",
            ),
        ]
    )
    service = ExperienceDurationService()
    assert service.compute(profile) is None


def test_missing_end_date_returns_none():
    profile = CandidateProfile(
        experience=[
            ExperienceItem(
                title="Role",
                company="Corp",
                start_date="2022.06",
                end_date=None,
            ),
        ]
    )
    service = ExperienceDurationService()
    assert service.compute(profile) is None


def test_total_years_rounds_down():
    profile = CandidateProfile(
        experience=[
            ExperienceItem(
                title="Role",
                company="Corp",
                start_date="2022.01",
                end_date="2023.06",
            ),
        ]
    )
    service = ExperienceDurationService(today=date(2024, 1, 1))
    assert service.total_years(profile) == 1
