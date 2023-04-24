import torch

# 從 PyTorch Hub 下載 YOLOv5s 預訓練模型，可選用的模型有 yolov5s, yolov5m, yolov5x 等
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 影像來源，支援檔案、路徑、PIL、OpenCV, NumPy, list 等
img = 'https://ultralytics.com/images/zidane.jpg'

# 進行物件偵測
results = model(img)

# 顯示結果摘要
results.print()
# 顯示結果圖片
results.show()

# 儲存結果圖片
results.save()
results.pandas().xyxy[0]