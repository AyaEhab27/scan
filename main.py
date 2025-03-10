from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
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

# تحديد مسار Tesseract (غير مطلوب في Docker إذا تم تثبيته بشكل صحيح)
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

    # تحويل الصورة إلى التدرج الرمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # إزالة الضوضاء باستخدام GaussianBlur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # زيادة التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # تطبيق العتبة التكيفية
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    return Image.fromarray(processed_img)

def clean_text(text):
    # الاحتفاظ فقط بالحروف العربية والإنجليزية والأرقام والمسافات
    cleaned_text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FFA-Za-z0-9 ]', '', text)
    return cleaned_text.strip()

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        
        img = Image.open(io.BytesIO(image_data))

        processed_img = preprocess_image(img)

        # تحسين إعدادات Tesseract
        custom_config = r'--psm 11 --oem 1'  # استخدام LSTM للتعرف على النصوص العربية
        text = pytesseract.image_to_string(processed_img, lang="ara+eng", config=custom_config)

        final_text = clean_text(text)

        return {"extracted_text": final_text}
    except Exception as e:
        return {"error": str(e)}
