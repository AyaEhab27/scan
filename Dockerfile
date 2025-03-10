# استخدام صورة أساسية تحتوي على Python
FROM python:3.9-slim

# تثبيت التبعيات النظامية (Tesseract ومكتباته)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    wget \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد ملفات اللغة
RUN mkdir -p /usr/share/tesseract-ocr/tessdata

# تحميل ملفات اللغة يدويًا (ara.traineddata و eng.traineddata)
RUN wget -O /usr/share/tesseract-ocr/tessdata/ara.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata && \
    wget -O /usr/share/tesseract-ocr/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# تعيين مسار ملفات اللغة (TESSDATA_PREFIX)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

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
