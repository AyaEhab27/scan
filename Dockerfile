FROM python:3.11

RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-ara

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
