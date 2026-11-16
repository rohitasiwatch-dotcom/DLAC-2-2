
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
        classes=[2, 3, 5, 7],
        verbose=False
    )

    vehicle_count = len(results[0].boxes)

    waiting_time = vehicle_count * 5

    output = results[0].plot()

    cv2.putText(
        output,
        f"Vehicles: {vehicle_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    cv2.putText(
        output,
        f"Estimated Waiting Time: {waiting_time} sec",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
    )

    cv2.imshow("Traffic Waiting Time", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
