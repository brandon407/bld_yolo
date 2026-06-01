from ultralytics.models import YOLO
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/v12/yolov12.yaml')
    # model = YOLO(model='ultralytics/cfg/models/11/yolo11l.yaml')
    model.train(data='./datasets/data.yaml', epochs=300, batch=1, device='0', imgsz=640, workers=1, cache=False,
                amp=True, mosaic=False, project='run/train', name='exp',)


