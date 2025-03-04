from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "OCR API is running!"}

@app.post("/ocr/")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        text = pytesseract.image_to_string(img)

        return {"extracted_text": text}
    except Exception as e:
        return {"error": str(e)}