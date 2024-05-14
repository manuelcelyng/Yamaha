from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2

from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("./model/best.pt")
results = model.track(source='./videos/01.mp4', conf=0.3, iou=0.5, show=True)