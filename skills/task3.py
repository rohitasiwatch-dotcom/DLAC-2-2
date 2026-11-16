
import cv2

img = cv2.imread("traffic.jpg")

if img is None:
    print("Image not found!")
else:
    cv2.imshow("Traffic Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
