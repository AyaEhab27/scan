from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np

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
    
    gray = cv2.equalizeHist(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return binary

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
        
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي َُِّّْ"
        
        custom_config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={allowed_chars}'
        text = pytesseract.image_to_string(processed_img, lang="ara+eng", config=custom_config)
        
        return {"extracted_text": text.strip()}
    except Exception as e:
        return {"error": str(e)}
