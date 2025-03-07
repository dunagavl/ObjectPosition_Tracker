from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Camera width
cap.set(4, 720) # Camera height

model = YOLO('../Yolo-Weights/yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # OpenCV bounding box method
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #CVZone bounding box method
            w,h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox)

            # Find the confidence of the object detection
            conf = math.floor(box.conf[0]*100)/100

            

    cv2.imshow("Image", img)
    cv2.waitKey(1)
