from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
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
else:
    raise Exception("Tesseract is not installed or not found in the system path.")

def preprocess_image(img):
    img = img.convert('L')
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        
        img = preprocess_image(img)
        
        text = pytesseract.image_to_string(img, lang="ara+eng")

        return {"extracted_text": text.strip()}
    except Exception as e:
        return {"error": str(e)}
