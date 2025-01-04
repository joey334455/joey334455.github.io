from ultralytics import YOLO
import cv2

# 加載 YOLO 模型
model = YOLO('yolov8n.pt')  # 使用 Nano 模型（輕量快速）

# 開啟攝影機（0 表示默認攝影機）
cap = cv2.VideoCapture(0)

# 檢查攝影機是否成功打開
if not cap.isOpened():
    print("無法啟動攝影機！")
    exit()

# 獲取攝影機畫面尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"攝影機畫面大小：{frame_width}x{frame_height}")

# 打開檔案準備寫入偵測結果
with open("筆電鏡頭辨識結果.txt", "w", encoding="utf-8") as file:
    file.write("YOLO 偵測結果:\n")

    # 即時處理攝影機畫面
    while True:
        ret, frame = cap.read()  # 從攝影機捕獲畫面
        if not ret:
            print("無法讀取攝影機畫面！")
            break

        # 使用 YOLO 模型進行推理
        results = model(frame)

        # 獲取帶檢測結果的畫面
        annotated_frame = results[0].plot()

        # 將偵測到的物件資訊寫入檔案
        for box in results[0].boxes:  # 遍歷所有檢測框
            cls = box.cls[0]  # 物件類別
            confidence = box.conf[0]  # 偵測信心
            xyxy = box.xyxy[0]  # 偵測框位置（左上與右下座標）

            # 格式化偵測結果
            result_str = f"類別: {cls}, 信心: {confidence:.2f}, 座標: {xyxy.tolist()}\n"
            file.write(result_str)

        # 顯示畫面
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 釋放攝影機和關閉視窗
cap.release()
cv2.destroyAllWindows()
