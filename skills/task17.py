
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("traffic.mp4")

previous_positions = {}

# Demo expected direction: vehicles move towards the right
EXPECTED_DIRECTION = "RIGHT"

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

    output = results[0].plot()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):
            x, y, w, h = box
            current_x = int(x)

            if track_id in previous_positions:
                old_x = previous_positions[track_id]

                movement = current_x - old_x

                if movement > 5:
                    direction = "RIGHT"
                elif movement < -5:
                    direction = "LEFT"
                else:
                    direction = "STRAIGHT / SLOW"

                if direction == "LEFT" and EXPECTED_DIRECTION == "RIGHT":
                    label = "CHECK TURN"
                else:
                    label = direction

                cv2.putText(
                    output,
                    f"ID {track_id}: {label}",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

            previous_positions[track_id] = current_x

    cv2.imshow("Wrong Turn Demonstration", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
