#!/bin/bash

apt-get update && apt-get install -y tesseract-ocr

exec uvicorn main:app --host 0.0.0.0 --port 8000
