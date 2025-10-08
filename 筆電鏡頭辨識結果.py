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

with open("筆電鏡頭辨識結果.txt", "w", encoding="utf-8") as file:
    file.write("YOLO 偵測結果:\n")

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("無法讀取攝影機畫面！")
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        for box in results[0].boxes:  
            cls = box.cls[0]  
            confidence = box.conf[0]  
            xyxy = box.xyxy[0]  

            result_str = f"類別: {cls}, 信心: {confidence:.2f}, 座標: {xyxy.tolist()}\n"
            file.write(result_str)

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
