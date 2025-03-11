from fastapi import FastAPI, UploadFile, File, HTTPException
import pytesseract
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
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

def preprocess_image(image):
    if len(image.shape) == 2:
        gray = image
    else: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def clean_text(text):
    allowed_chars = r"[^a-zA-Z0-9\u0600-\u06FF\s]"
    cleaned_text = re.sub(allowed_chars, "", text)
    return cleaned_text.strip()

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        
        img_cv = np.array(img)
        
        processed_img = preprocess_image(img_cv)
        
        custom_config = f'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(processed_img, lang="ara+eng", config=custom_config)
        
        cleaned_text = clean_text(text)
        
        return {"extracted_text": cleaned_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
