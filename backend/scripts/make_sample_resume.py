"""Generate a small sample resume PDF for manually testing the upload flow."""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tests.utils import make_pdf

RESUME_LINES = [
    "Jane Doe",
    "Senior Software Engineer",
    "jane.doe@example.com | (555) 123-4567 | San Francisco, CA",
    "",
    "Skills",
    "Python, JavaScript, TypeScript, React, FastAPI, SQL, PostgreSQL, Docker, AWS, Git",
    "",
    "Experience",
    "Senior Software Engineer - TechCorp (2020-2023)",
    "- Built and operated REST APIs with FastAPI and PostgreSQL serving 2M+ requests/day",
    "- Deployed containerized services with Docker on AWS",
    "- Mentored two junior engineers",
    "",
    "Software Engineer - StartupXYZ (2018-2020)",
    "- Developed React frontend components used by 100k+ monthly active users",
    "- Wrote unit tests with pytest and integration tests with Playwright",
    "",
    "Education",
    "B.S. Computer Science - State University (2018)",
]


def main() -> None:
    output = BACKEND_DIR / "sample_data" / "sample_resume.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(make_pdf(RESUME_LINES))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
