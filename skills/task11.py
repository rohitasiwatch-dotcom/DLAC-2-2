import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame, classes=[2, 3, 5, 7], verbose=False)

    count = len(results[0].boxes)

    if count <= 5:
        status = "Normal Traffic"
    elif count <= 15:
        status = "Heavy Traffic"
    else:
        status = "CONGESTION ALERT!"

    output = results[0].plot()

    cv2.putText(output, f"Vehicles: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(output, status,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Traffic Congestion Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()