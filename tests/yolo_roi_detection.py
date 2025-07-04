
# This is a placeholder for YOLOv8-based ROI detection
# You will need to install ultralytics and load a trained model

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Load lightweight model

img = cv2.imread("scene.jpg")
results = model(img)

for r in results:
    boxes = r.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("YOLO ROI Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
