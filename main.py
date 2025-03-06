from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tesseract_path = "/usr/bin/tesseract"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        text = pytesseract.image_to_string(img, lang="ara+eng")

        return {"extracted_text": text.strip()}
    except Exception as e:
        return {"error": str(e)}
