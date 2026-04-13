import numpy as np

from adapter_core import YOLOAdapter
from yolo_tensor import YOLOPoseOut


class YOLOPoseAdapter(YOLOAdapter):
    def __init__(self, model_path="yolo26n-pose.pt", conf=0.25, iou=0.45, device="cpu"):
        self.keypoints_xy = None
        self.keypoints_confs = None
        super().__init__(model_path=model_path, conf=conf, iou=iou, device=device)


    def get_output(self, frame):
        results = self._run_inference(frame)
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            bboxes = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(np.int32)
            ids = boxes.id.cpu().numpy().astype(np.int32).tolist() if boxes.is_track else None

            if results[0].keypoints:
                keypoints = results[0].keypoints
                self.keypoints = keypoints.xy.cpu().numpy()
                self.keypoints_confs = keypoints.conf.cpu().numpy()
            else:
                keypoints = None
                self.keypoints = None
                self.keypoints_confs = None
            return bboxes, classes, ids
        
        self.keypoints = []
        self.keypoints_confs = []
        return [], [], []

        
    def get_output_tensor(self, frame) -> YOLOPoseOut:
        results = self._run_inference(frame)
        img_shape = frame.shape[:2]
        
        if results and results[0].boxes is not None:
            data = results[0].boxes.data.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None
            return YOLOPoseOut(data=data, orig_shape=img_shape, keypoints_data=keypoints)
        
        return YOLOPoseOut(data=None, orig_shape=img_shape, keypoints_data=None)


    


            


