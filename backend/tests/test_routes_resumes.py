def test_upload_resume_success(client, sample_resume_pdf):
    response = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.pdf", sample_resume_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "resume.pdf"
    assert body["page_count"] == 1
    assert "Jane Doe" in body["extracted_text"]
    assert body["character_count"] == len(body["extracted_text"])


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_corrupt_pdf(client):
    response = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.pdf", b"%PDF-1.4 broken", "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_rejects_non_pdf_bytes(client):
    response = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.pdf", b"PK\x03\x04 this is a zip file", "application/pdf")},
    )
    assert response.status_code == 415
    assert "valid PDF" in response.json()["detail"]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
