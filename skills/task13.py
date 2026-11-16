import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

previous_positions = {}

PIXELS_PER_METER = 10

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):
            x, y, w, h = box

            if track_id in previous_positions:
                old_x, old_y = previous_positions[track_id]

                distance = ((x-old_x)**2 + (y-old_y)**2)**0.5
                speed = distance * fps / PIXELS_PER_METER

                cv2.putText(
                    frame,
                    f"ID {track_id}: {speed:.1f} m/s (approx)",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

            previous_positions[track_id] = (x, y)

    output = results[0].plot()
    cv2.imshow("Vehicle Speed Estimation", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()