import cv2

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("Video not found!")
else:
    while True:
        ret, frame = video.read()

        if not ret:
            break

        cv2.imshow("Traffic Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()