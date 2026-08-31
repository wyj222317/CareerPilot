def make_pdf(lines: list[str]) -> bytes:
    """Build a minimal single-page PDF containing the given text lines."""

    def escape(value: str) -> str:
        return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    ops = ["BT", "/F1 11 Tf"]
    if lines:
        ops.append("72 720 Td")
        for i, line in enumerate(lines):
            if i > 0:
                ops.append("0 -18 Td")
            ops.append(f"({escape(line)}) Tj")
    ops.append("ET")
    content = "\n".join(ops).encode("latin-1")

    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        4: b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in objects.items():
        offsets[number] = len(pdf)
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_position = len(pdf)
    pdf += b"xref\n0 6\n0000000000 65535 f \n"
    for number in range(1, 6):
        pdf += f"{offsets[number]:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n".encode()
    return bytes(pdf)
