from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image, ImageEnhance
import io
import cv2
import numpy as np
import os
import re

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

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

def preprocess_image(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    pil_img = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(2)  

    enhanced_img_np = np.array(enhanced_img)

    processed_img = cv2.adaptiveThreshold(
        enhanced_img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    return Image.fromarray(processed_img)

def clean_text(text):
    cleaned_text = re.sub(r'[^أ-يA-Za-z0-9 ]', '', text)
    return cleaned_text.strip()

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        
        img = Image.open(io.BytesIO(image_data))

        processed_img = preprocess_image(img)

        custom_config = r'--psm 11 --oem 1'  
        text = pytesseract.image_to_string(processed_img, lang="ara+eng", config=custom_config)

        final_text = clean_text(text)

        return {"extracted_text": final_text}
    except Exception as e:
        return {"error": str(e)}
