from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('./static/best_1.pt')

# Define path to the image file
source = 'static/temp/test.jpeg'

# Run inference on the source
results = model(source, save=True)
