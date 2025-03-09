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
    # تحويل الصورة إلى تدرج رمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين باستخدام CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # تقليل الضوضاء باستخدام Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # استخدام Adaptive Thresholding لتحويل الصورة إلى ثنائية (أسود وأبيض)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return thresh

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        # قراءة الصورة من الملف المرفوع
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # تحسين جودة الصورة
        processed_image = preprocess_image(img)
        
        # تحويل الصورة المعالجة إلى صيغة PIL لاستخدامها مع pytesseract
        pil_image = Image.fromarray(processed_image)
        
        # استخراج النص باستخدام Tesseract مع تحسين الإعدادات
        custom_config = r'--psm 6 -l ara+eng'  # PSM 6 يفترض أن الصورة تحتوي على فقرة واحدة من النص
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        return {"extracted_text": text.strip()}
    except Exception as e:
        return {"error": str(e)}
