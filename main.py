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

# إضافة CORS middleware للسماح بطلبات من أي مصدر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# التحقق من تثبيت Tesseract
tesseract_path = "/usr/bin/tesseract"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise Exception("Tesseract is not installed or not found in the system path.")

# تحسين جودة الصورة
def preprocess_image(image):
    # إذا كانت الصورة رمادية (قناة واحدة)
    if len(image.shape) == 2:
        gray = image
    else:  # إذا كانت الصورة ملونة (3 قنوات)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين باستخدام CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # تقليل الضوضاء باستخدام مرشح غاوسي
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # تحويل الصورة إلى صورة ثنائية (أبيض وأسود)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

# تحديد اللغة بناءً على النص
def detect_language(text):
    # استخدام تعبيرات منتظمة للكشف عن اللغة
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')  # نطاق الحروف العربية
    if arabic_pattern.search(text):
        return "ara"  # اللغة العربية
    else:
        return "eng"  # اللغة الإنجليزية

# إزالة الرموز والعلامات غير المرغوب فيها
def clean_text(text):
    # السماح فقط بالحروف العربية، الإنجليزية، الأرقام، والمسافات
    allowed_chars = r"[^a-zA-Z0-9\u0600-\u06FF\s]"
    cleaned_text = re.sub(allowed_chars, "", text)
    return cleaned_text.strip()

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        # قراءة الصورة
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        
        # تحويل الصورة إلى مصفوفة numpy
        img_cv = np.array(img)
        
        # تحسين جودة الصورة
        processed_img = preprocess_image(img_cv)
        
        # استخراج النص الأولي
        initial_text = pytesseract.image_to_string(processed_img, lang="ara+eng")
        
        # تحديد اللغة بناءً على النص المستخرج
        language = detect_language(initial_text)
        
        # استخراج النص مرة أخرى باستخدام اللغة المحددة
        final_text = pytesseract.image_to_string(processed_img, lang=language)
        
        # تنظيف النص من الرموز غير المرغوب فيها
        cleaned_text = clean_text(final_text)
        
        return {"extracted_text": cleaned_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
