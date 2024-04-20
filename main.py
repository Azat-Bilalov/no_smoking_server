from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
from ultralytics import YOLO

model = YOLO("./static/yolov8n-pose.pt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File()):
    with open(f"static/temp/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())

    source = "static/temp/" + file.filename
    model(source, save=True, project="static", name="results", exist_ok=True)
    print()

    os.remove(source)

    return FileResponse("static/results/" + file.filename)
