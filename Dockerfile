# استخدام صورة أساسية تحتوي على Python
FROM python:3.9-slim

# تثبيت التبعيات النظامية (Tesseract ومكتباته)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# تعيين مسار ملفات اللغة (TESSDATA_PREFIX)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# إنشاء مجلد العمل
WORKDIR /app

# نسخ ملفات المشروع إلى مجلد العمل
COPY . .

# تثبيت تبعيات Python
RUN pip install --no-cache-dir -r requirements.txt

# تعيين المنفذ الذي سيعمل عليه التطبيق
EXPOSE 8000

# تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
