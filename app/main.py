from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import analyze
from .parser import UnsupportedFileError, extract

MAX_BYTES = 5 * 1024 * 1024  # 5 MB
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="ATS Resume Checker", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(..., description="Resume as PDF, DOCX or TXT"),
    job_description: Optional[str] = Form(None, description="Optional job description for keyword matching"),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 5 MB.")
    try:
        parsed = extract(file.filename or "", data)
    except UnsupportedFileError as e:
        raise HTTPException(415, str(e))
    except Exception:
        raise HTTPException(422, "We couldn't read this file. It may be corrupted or password-protected.")
    return analyze(parsed, job_description)
