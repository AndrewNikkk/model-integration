import numpy as np

from yolo26 import YOLOAdapter
from data_formats.yolo import YOLOPoseOut


class YOLOPoseAdapter(YOLOAdapter):
    def __init__(self, model_path='yolo26n.pt', track: bool = True, image_size: int | tuple[int, int] = 640,
                classes: list[int] | None = None, conf=0.25, iou=0.45, device='cuda'):
        """
        Модель YOLO с отслеживанием ключевых точек скелета.
        
        :param model_path: Имя/путь весов.
        :param track: Отслеживать ли объекты.
        :param image_size: Размер изображения для пересчёта координат [height, width].
        :param classes: Какие классы искать.
        :param conf: Порог точности.
        :param iou: Порог iou.
        :param device: Устройство для распознавания.
        """
        self.keypoints = None
        self.keypoints_confs = None
        super().__init__(model_path=model_path, conf=conf, iou=iou, device=device)


    def infer(self, frame: np.ndarray):
        results = self._run_inference(frame=frame)
        box_data = results[0].boxes.data.cpu().numpy() if len(results) != 0 else None
        keypoints_data = results[0].keypoints.data.cpu().numpy() if len(results) != 0 else None
        return YOLOPoseOut(frame.shape[:2], box_data, keypoints_data, 17)

    
    def get_output(self, frame):
        dto_bbk: YOLOPoseOut = self.infer(frame)
        self.keypoints = dto_bbk.keypoints.xy
        self.keypoints_data = dto_bbk.keypoints_data
        return dto_bbk.old_format

    


            


