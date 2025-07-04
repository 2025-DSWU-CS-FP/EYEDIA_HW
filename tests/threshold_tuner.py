
import cv2

img = cv2.imread("./eye_img/eye.png", cv2.IMREAD_GRAYSCALE)

def nothing(x):
    pass

cv2.namedWindow("Tuner")
cv2.createTrackbar("Block Size", "Tuner", 11, 50, nothing)
cv2.createTrackbar("C Value", "Tuner", 3, 20, nothing)

while True:
    blk = cv2.getTrackbarPos("Block Size", "Tuner")
    if blk % 2 == 0: blk += 1
    if blk < 3: blk = 3
    c = cv2.getTrackbarPos("C Value", "Tuner")

    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blk, c)
    cv2.imshow("Tuner", th)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
