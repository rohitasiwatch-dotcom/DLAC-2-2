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
        density = "Low"
    elif count <= 15:
        density = "Medium"
    else:
        density = "High"

    output = results[0].plot()

    cv2.putText(output, f"Vehicles: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(output, f"Traffic Density: {density}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Traffic Density", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()