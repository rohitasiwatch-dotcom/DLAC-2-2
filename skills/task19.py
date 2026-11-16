
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("traffic.mp4")

# Restricted area coordinates
x1, y1 = 200, 200
x2, y2 = 600, 450

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    output = results[0].plot()

    cv2.rectangle(
        output, (x1, y1), (x2, y2),
        (0, 0, 255), 2
    )

    for box in results[0].boxes:
        x, y, w, h = box.xywh[0].cpu().numpy()

        center_x = int(x)
        center_y = int(y)

        if x1 < center_x < x2 and y1 < center_y < y2:
            cv2.putText(
                output,
                "RESTRICTED AREA!",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )

    cv2.imshow("Restricted Area Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
