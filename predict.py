from ultralytics.models import YOLO


if __name__ == '__main__':
    model = YOLO(model='run/train/exp2/weights/best.pt')
    model.predict(source='run/train/exp2/train_batch0.jpg', device='0', imgsz=640, project='runs/detect/', name='exp',save=True)
