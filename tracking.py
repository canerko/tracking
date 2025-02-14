from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolo11n.pt")
#results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
results = model.track(source="../people-counter/giris.mp4", conf=0.3, iou=0.5, show=True)
