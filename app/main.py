from fastapi import FastAPI, UploadFile, File, Form
from .utils import extract_text_from_pdf
from .agent import agent

app = FastAPI()

@app.post("/analyze-pdf")
async def analyze_pdf(
    resume: UploadFile = File(...),
    goal: str = Form(...)
):
    pdf_bytes = await resume.read()
    resume_text = extract_text_from_pdf(pdf_bytes)

    result = await agent.run(
        f"Resume:\n{resume_text}\n\nCareer Goal:\n{goal}"
    )

    return result.data

