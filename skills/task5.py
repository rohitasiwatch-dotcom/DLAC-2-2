import cv2

video = cv2.VideoCapture("traffic.mp4")

count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    if count % 30 == 0:
        cv2.imwrite(f"frame_{count}.jpg", frame)

    count += 1

video.release()

print("Frames extracted successfully!")