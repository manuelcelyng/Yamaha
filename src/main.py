import torch
import numpy as numpy
import cv2
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, video_path, yolo_path):
        self.capture_index =  video_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)

        self.model = self.load_model(yolo_path)

    def load_model(self, yolo_path):
        model =  YOLO(yolo_path)
        model.fuse()
        return model
    
    def predict(self, frame):
        results = self.model(frame, conf=0.75)
        return results
    

    def plot_bboxes(self,results,frame):
        xyxys = [] # los puntos del bounding box
        confidences = []  # Estas son las probabilidades
        class_ids = []  # el id de cada una de las clases

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        
        return result.plot(), xyxys, confidences, class_ids
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Assuming fps is available
        video_writer = cv2.VideoWriter("ultralytics.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 640))
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (640, 640))
                results = self.predict(frame)
                frame_with_boxes, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
                cv2.imshow('Object Detection', frame_with_boxes)
                cv2.resizeWindow('Object Detection', 640, 640)
                video_writer.write(frame_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            print("Video frame esta vacio, o el procesamiento del video ha sido completado.")
            break



video_path = "./videos/01.mp4"  # Video path
yolo_model_path = "./model/best.pt"  # Path del modelo
# Creo el objeto con el path del video y el del modelo.
detector = ObjectDetection(video_path,yolo_model_path)
detector()