
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("traffic.mp4")

line_y = 300
previous_positions = {}

entering = 0
leaving = 0

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

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y),
             (255, 0, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):
            x, y, w, h = box
            current_y = int(y)

            if track_id in previous_positions:
                old_y = previous_positions[track_id]

                if old_y < line_y and current_y >= line_y:
                    entering += 1

                elif old_y > line_y and current_y <= line_y:
                    leaving += 1

            previous_positions[track_id] = current_y

    output = results[0].plot()

    cv2.line(output, (0, line_y),
             (output.shape[1], line_y), (255, 0, 0), 2)

    cv2.putText(output, f"Entering: {entering}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(output, f"Leaving: {leaving}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.imshow("Vehicle Entry Exit Counter", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
