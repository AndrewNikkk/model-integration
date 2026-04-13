from typing import Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from yolo_tensor import YOLOBaseOut, YOLOPoseOut


class AdapterMixin(ABC):
    @abstractmethod
    def get_output(self, frame: np.ndarray) -> Tuple[List[List[float]], List[int], List[int]]:
        pass

    def get_output_tensor(self, frame: np.ndarray) -> YOLOBaseOut | YOLOPoseOut:
        pass

    def set_mode(self, mode: str):
        pass


class YOLOAdapter(AdapterMixin):
    def __init__(self, model_path='yolo26n.pt', conf=0.25, iou=0.45, device='cpu'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_thres = conf
        self.iou_thres = iou
        self.device = device
        self.mode = 'predict'
        self.img_shape = None


    def _run_inference(self, frame):
        if self.mode == 'predict':
            return self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres,
                                        device=self.device, verbose=False)
        elif self.mode == 'track':
            return self.model.track(
                frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device,
                persist=True,
                verbose=False
            )
            

    def get_output_tensor(self, frame) -> YOLOBaseOut:
        results = self._run_inference(frame=frame)

        self.img_shape = frame.shape[:2]

        if results and results[0].boxes is not None:
            data = results[0].boxes.data.cpu().numpy()
            return YOLOBaseOut(data=data, orig_shape=self.img_shape)
        else:
            return YOLOBaseOut(data=None, orig_shape=self.img_shape)
        

    def get_output(self, frame):
        results = self._run_inference(frame=frame)

        self.img_shape = frame.shape[:2]

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            bboxes = boxes.xyxy.cpu().numpy().tolist()
            classes = boxes.cls.cpu().numpy().astype(np.int32).tolist()
            ids = boxes.id.cpu().numpy().astype(np.int32).tolist() if boxes.is_track else None

            return bboxes, classes, ids
        
        else:
            return [], [], []

            
    def set_mode(self, mode):
        if mode not in ('predict', 'track'):
            raise ValueError("Mode must be 'predict' or 'track'")
        if mode == 'predict' and self.mode == 'track':
            self._reset_tracks()
        self.mode = mode

    
    def _reset_tracks(self):
        if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
            self.model.predictor.trackers.clear()
