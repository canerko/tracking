import cv2
from ultralytics import YOLO

# Load the YOLOv8n model (or a suitable one for person detection and tracking)
model = YOLO("yolo11n.pt")
#model = YOLO('yolov8n.pt')  # Or a larger model if needed: yolov8s.pt, yolov8m.pt, etc.

# Open the video file
video_path = "../people-counter/giris.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform object detection and tracking
        results = model.track(frame, persist=True)

        # Iterate through the detected/tracked objects
        for result in results:
            boxes = result.boxes
            track_ids = result.boxes.id  # Get the track IDs

            if track_ids is not None:  # Check if track IDs exist (important!)
                for box, track_id in zip(boxes, track_ids):  # Iterate through boxes and track IDs
                    class_id = int(box.cls)
                    if class_id == 0:  # Check if it's a person (class 0 in COCO)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                        conf = float(box.conf)
                        label = f'Person {int(track_id)} ({conf:.2f})' # Include Track ID and Confidence
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Display the frame with person detections and tracking
        cv2.imshow("Person Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()