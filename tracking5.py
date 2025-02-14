import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolo11n.pt")  # veya yolov8n.pt, yolov8s.pt, vb.

# Video dosyasını aç
video_path = "../people-counter/giris.mp4"
cap = cv2.VideoCapture(video_path)

# Logo dosyasını oku
logo_path = "logo.png"
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

# Logo'nun boyutlarını ayarla (isteğe bağlı)
new_logo_height = 100
new_logo_width = int(3.82 * new_logo_height)
logo = cv2.resize(logo, (new_logo_width, new_logo_height))

# Çizgi koordinatları (ekranın ortasında dikey çizgi)
line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)  # Ortada dikey çizgi için
entered_count = 0
exited_count = 0
tracked_persons = {}  # Takip edilen kişilerin bilgilerini saklamak için sözlük

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        for result in results:
            boxes = result.boxes
            track_ids = result.boxes.id

            if track_ids is not None:
                for box, track_id in zip(boxes, track_ids):
                    class_id = int(box.cls)
                    if class_id == 0:  # Kişi ise
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kutuyu çiz

                        conf = float(box.conf)
                        label = f'Person {int(track_id)} ({conf:.2f})'
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        #print(type(track_id))
                        #for person in tracked_persons:
                        #    print(f"tracked_person: {person} and {tracked_persons[person]}")
                            

                        # Çizgi geçiş kontrolü (Dikey çizgi için x koordinatları kontrol ediliyor)
                        if str(track_id) not in tracked_persons:
                            tracked_persons[str(track_id)] = center_x  # İlk konumu kaydet
                            print(f"ID:{track_id} x:{center_x} y:{center_y} tracked_persons[{str(track_id)}]:{tracked_persons[str(track_id)]}")
                        else:
                            previous_x = tracked_persons[str(track_id)]
                            if center_x > line_x and previous_x <= line_x:  # Çizgiyi soldan sağa geçti
                                entered_count += 1
                                print(f"ID:{track_id} girdi")
                            elif center_x < line_x and previous_x >= line_x:  # Çizgiyi sağdan sola geçti
                                exited_count += 1
                                print(f"ID:{track_id} cikti")
                            tracked_persons[str(track_id)] = center_x  # Yeni konumu güncelle
                            print(f"ID:{track_id} x:{center_x} y:{center_y} previous_x:{previous_x}")

        # Dikey çizgiyi çiz
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)  # Kırmızı dikey çizgi

        # Sayıları ekrana yazdır
        cv2.putText(frame, f"Giren: {entered_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cikan: {exited_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Logo ekleme (aynı kalıyor)
        frame_height, frame_width, _ = frame.shape
        logo_x = frame_width - logo.shape[1] - 15
        logo_y = 10
        #logo_y = frame_height - logo.shape[0]

        for c in range(0, 3):
            frame[logo_y:logo_y + logo.shape[0], logo_x:logo_x + logo.shape[1], c] = \
                logo[0:logo.shape[0], 0:logo.shape[1], c] * (logo[:, :, 3] / 255) + \
                frame[logo_y:logo_y + logo.shape[0], logo_x:logo_x + logo.shape[1], c] * (1.0 - logo[:, :, 3] / 255)

        cv2.imshow("Person Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()