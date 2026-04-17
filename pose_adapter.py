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
            boxes = results[0].boxes.cpu()

            bboxes = boxes.xyxy.numpy().astype(np.int32).tolist()
            classes = boxes.cls.numpy().astype(np.int32).tolist()
            ids = boxes.id.numpy().astype(np.int32).tolist() if boxes.is_track else None

            if results[0].keypoints is not None:
                keypoints = results[0].keypoints
                self.keypoints_xy = keypoints.xy.cpu().numpy().astype(np.int32).tolist()
                self.keypoints_confs = keypoints.conf.cpu().numpy().tolist()
            else:
                self.keypoints_xy = None
                self.keypoints_confs = None

            return bboxes, classes, ids
        
        self.keypoints_xy = None
        self.keypoints_confs = None
        return None, None, None

        
    def get_output_tensor(self, frame) -> YOLOPoseOut:
        results = self._run_inference(frame=frame)
        img_shape = frame.shape[:2]

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.cpu()

            if boxes.is_track:
                data = boxes.data.cpu().numpy()
                keypoints_data = results[0].keypoints.data.cpu() if results[0].keypoints is not None else None
                return YOLOPoseOut(data=data, orig_shape=img_shape, keypoints_data=keypoints_data)
            else:
                bboxes = boxes.xyxy
                classes = boxes.cls
                confs = boxes.conf
                keypoints = results[0].keypoints.data.cpu() if results[0].keypoints is not None else None
                ids = np.asarray([-1] * len(bboxes))
                return YOLOPoseOut.from_columns(xyxy=bboxes, ids=ids, confs=confs, classes=classes, keypoints=keypoints, orig_shape=img_shape)

        return YOLOPoseOut(data=None, orig_shape=img_shape, keypoints_data=None)


    


            


