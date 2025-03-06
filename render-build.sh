#!/bin/bash

apt-get update && apt-get install -y \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-ara

pip install --upgrade pip
pip install -r requirements.txt
