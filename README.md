# Installation

1. Install OpenCV (tested with version 4.10)

```
# brew install opencv
```

2. Download detection model/s (using a working Python env). For a list of available models check https://docs.ultralytics.com/tasks/detect/

```
# pip install ultralytics
# yolo export model=models/yolov8m.pt imgsz=640 format=onnx opset=12
```

3. Search for results within image

```
# cargo run -- -i=assets/people4.jpg -m=models/yolov8n.onnx
```
