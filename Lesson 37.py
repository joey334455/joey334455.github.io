from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt') 

video_path = 'C:\\Users\\user\\Downloads\\people.mp4' 
output_path = 'output_video.avi' 

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("無法打開影片文件！請檢查路徑。")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

print(f"影片資訊: {fps} FPS, 寬度: {frame_width}, 高度: {frame_height}, 幀數: {frame_count}")

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    frame_index += 1
    print(f"處理幀 {frame_index}/{frame_count}", end="\r")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"處理完成！影片已保存到: {output_path}")
