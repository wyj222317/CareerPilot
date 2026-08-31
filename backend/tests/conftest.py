import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.utils import make_pdf


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_resume_pdf() -> bytes:
    return make_pdf([
        "Jane Doe",
        "Senior Software Engineer",
        "jane.doe@example.com | (555) 123-4567 | San Francisco, CA",
        "Skills: Python, JavaScript, React, FastAPI, SQL, Docker, AWS",
        "Experience:",
        "Senior Software Engineer - TechCorp (2020-2023)",
        "Built REST APIs with FastAPI and PostgreSQL.",
        "Software Engineer - StartupXYZ (2018-2020)",
        "Developed React frontend components.",
        "Education: B.S. Computer Science, State University",
    ])
