
import cv2

image = cv2.imread("traffic.jpg")

if image is None:
    print("Image not found!")
    exit()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

red_lower1 = (0, 100, 100)
red_upper1 = (10, 255, 255)

red_lower2 = (160, 100, 100)
red_upper2 = (180, 255, 255)

yellow_lower = (15, 100, 100)
yellow_upper = (35, 255, 255)

green_lower = (40, 50, 50)
green_upper = (90, 255, 255)

red_mask = cv2.bitwise_or(
    cv2.inRange(hsv, red_lower1, red_upper1),
    cv2.inRange(hsv, red_lower2, red_upper2)
)

yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
green_mask = cv2.inRange(hsv, green_lower, green_upper)

red_count = cv2.countNonZero(red_mask)
yellow_count = cv2.countNonZero(yellow_mask)
green_count = cv2.countNonZero(green_mask)

counts = {
    "RED": red_count,
    "YELLOW": yellow_count,
    "GREEN": green_count
}

status = max(counts, key=counts.get)

if counts[status] < 20:
    status = "SIGNAL NOT DETECTED"

print("Traffic Signal Status:", status)

cv2.putText(
    image,
    status,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Traffic Signal Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
