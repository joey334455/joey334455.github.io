from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法啟動攝影機！")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"攝影機畫面大小：{frame_width}x{frame_height}")

while True:
    ret, frame = cap.read() 
    if not ret:
        print("無法讀取攝影機畫面！")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
