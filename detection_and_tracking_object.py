import cv2
from tracker import *
output = cv2.VideoWriter('aya.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (250, 250))
tracker = EuclideanDistTracker()
cap = cv2.VideoCapture("highway.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)
while True:
    ret, frame = cap.read()
    #height, width, _ = frame.shape
    #roi = frame[100: 833, 10: 900]
    roi = frame[340: 720,500: 800]
    mask = object_detector.apply(roi)
    #gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    #blur=cv2.blur(mask,(3,3),0)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append([x, y, w, h])
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    output.write(frame)
    key = cv2.waitKey(30)
    # if key == 27:
    # break
    if key == ord('q'):
        cv2.waitKey(-1)
        break

output.release()
cap.release()
cv2.destroyAllWindows()
