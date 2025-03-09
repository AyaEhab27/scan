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

# تفعيل CORS للسماح بالاتصال من أي جهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحديد مسار Tesseract
tesseract_path = "/usr/bin/tesseract"  # إذا كنت على Windows غيّره إلى مسار التثبيت
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise Exception("Tesseract is not installed or not found in the system path.")

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

# دالة لتحسين الصورة قبل استخراج النص
def preprocess_image(image):
    # تحويل الصورة إلى numpy array
    img = np.array(image)

    # تحويل الصورة إلى تدرج الرمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # تطبيق Adaptive Thresholding لتحسين النص
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    return Image.fromarray(processed_img)

# دالة لتنظيف النص المستخرج
def clean_text(text):
    # الاحتفاظ فقط بالحروف العربية والإنجليزية والأرقام والمسافات
    cleaned_text = re.sub(r'[^أ-يA-Za-z0-9 ]', '', text)
    return cleaned_text.strip()

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        # قراءة بيانات الصورة
        image_data = await image.read()
        
        # فتح الصورة باستخدام PIL
        img = Image.open(io.BytesIO(image_data))

        # تحسين جودة الصورة قبل المعالجة
        processed_img = preprocess_image(img)

        # استخراج النص باستخدام Tesseract مع تمرير إعدادات إضافية
        custom_config = r'--psm 6 --oem 3'  # تحسين التعرف على النصوص المتصلة
        text = pytesseract.image_to_string(processed_img, lang="ara+eng", config=custom_config)

        # تنظيف النص من الرموز غير المرغوبة
        final_text = clean_text(text)

        return {"extracted_text": final_text}
    except Exception as e:
        return {"error": str(e)}
