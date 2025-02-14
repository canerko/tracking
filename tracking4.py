import cv2
from ultralytics import YOLO

# Load the YOLOv8n model (or a suitable one for person detection and tracking)
model = YOLO("yolo11n.pt")
#model = YOLO('yolov8n.pt')  # Or a larger model if needed: yolov8s.pt, yolov8m.pt, etc.

# Open the video file
video_path = "../people-counter/giris.mp4" 
cap = cv2.VideoCapture(video_path)

# Logo dosyasını oku
logo_path = "logo.png"  
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # Şeffaflık için IMREAD_UNCHANGED kullanın

# Logo'nun boyutlarını küçült (isteğe bağlı)
new_logo_width = 381    #
new_logo_height = 100   # 
logo = cv2.resize(logo, (new_logo_width, new_logo_height))

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

        # Logo'nun konumunu hesapla (sol alt köşe)
        frame_height, frame_width, _ = frame.shape
        logo_x = 15
        logo_y = 10
        #logo_y = frame_height - logo.shape[0]

        # Şeffaflığı kullanarak logoyu video çerçevesine ekle
        for c in range(0, 3):
            frame[logo_y:logo_y + logo.shape[0], logo_x:logo_x + logo.shape[1], c] = \
                logo[0:logo.shape[0], 0:logo.shape[1], c] * (logo[:, :, 3] / 255) + \
                frame[logo_y:logo_y + logo.shape[0], logo_x:logo_x + logo.shape[1], c] * (1.0 - logo[:, :, 3] / 255)

        # Display the frame with person detections and tracking
        cv2.imshow("Person Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()