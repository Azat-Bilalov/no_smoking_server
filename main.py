from math import sqrt
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

cigarettes_model = YOLO("./static/cigarettes.pt")
pose_model = YOLO("./static/pose.pt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["cigarette_exists"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def valid(result):
    '''Проверка наличия сигареты по позе и наличию сигареты на изображении'''
    lst = result[0].keypoints.xy.tolist()[0]
    dist_min_8_10 = sqrt(
        pow(abs(lst[8][0] - lst[10][0]), 2) + pow(abs(lst[8][1] - lst[10][1]), 2))
    dist_min_9_7 = sqrt(
        pow(abs(lst[9][0] - lst[7][0]), 2) + pow(abs(lst[9][1] - lst[7][1]), 2))
    dist_min_6_10 = sqrt(
        pow(abs(lst[6][0] - lst[10][0]), 2) + pow(abs(lst[6][1] - lst[10][1]), 2))
    dist_min_7_5 = sqrt(
        pow(abs(lst[7][0] - lst[5][0]), 2) + pow(abs(lst[7][1] - lst[5][1]), 2))

    clss = result[0].boxes.cls.tolist()
    for i in clss:
        if 'cigarette' == result[0].names[i]:
            return True

    return dist_min_8_10 > dist_min_6_10 or dist_min_9_7 > dist_min_7_5


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File()):
    with open(f"static/temp/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())

    source = "static/temp/" + file.filename

    if 'video' in file.content_type:
        cigarettes_result = cigarettes_model(
            source, project="static", save=True, exist_ok=True, name="results")

        filename = file.filename[:-4] + ".avi"
        export_filename = file.filename[:-4] + ".mp4"

        clip = VideoFileClip(source)
        clip.write_videofile("static/results/" + export_filename)

        source = "static/results/" + export_filename

        return FileResponse("static/results/" + export_filename)

    cigarettes_result = cigarettes_model(
        source, project="static", exist_ok=True)

    pose_result = pose_model(source, project="static", exist_ok=True)

    cigarettes_result[0].keypoints = pose_result[0].keypoints
    cigarette_exists = valid(cigarettes_result)
    cigarettes_result[0].save()

    os.remove(source)

    for filename in os.listdir("static/results"):
        os.remove("static/results/" + filename)
    os.rename('results_' + file.filename, "static/results/" + file.filename)

    return FileResponse("static/results/" + file.filename, headers={"cigarette_exists": str(cigarette_exists)})
