from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return thresh

def detect_language(image):
    try:
        osd = pytesseract.image_to_osd(image)
        language = osd.split("\n")[1].split(":")[1].strip()
        return "ara" if language == "Arabic" else "eng"
    except:
        return "eng"

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        processed_image = preprocess_image(img)
        
        pil_image = Image.fromarray(processed_image)
        
        language = detect_language(pil_image)
        
        custom_config = r'--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ء-ي '  # تجاهل الرموز والأشكال
        text = pytesseract.image_to_string(pil_image, config=custom_config, lang=language)
        
        return {"extracted_text": text.strip(), "detected_language": language}
    except Exception as e:
        return {"error": str(e)}
