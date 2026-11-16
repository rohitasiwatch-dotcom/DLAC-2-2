
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[0, 3],
        verbose=False
    )

    output = results[0].plot()

    persons = []
    motorcycles = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            persons.append((x1, y1, x2, y2))

        elif cls == 3:
            motorcycles.append((x1, y1, x2, y2))

    for mx1, my1, mx2, my2 in motorcycles:
        count = 0

        for px1, py1, px2, py2 in persons:
            if (mx1 - 30 <= (px1 + px2) // 2 <= mx2 + 30
                and my1 - 180 <= py2 <= my2 + 50):
                count += 1

        if count > 2:
            label = "MORE THAN 2 PERSONS!"
            color = (0, 0, 255)
        else:
            label = f"Persons: {count}"
            color = (0, 255, 0)

        cv2.putText(
            output, label, (mx1, my1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imshow("Two Wheeler Person Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
