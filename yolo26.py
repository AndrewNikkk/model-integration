from ultralytics import YOLO
from typing import Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from data_formats.yolo import YOLOBaseOut, YOLOPoseOut


class AdapterMixin(ABC):
    @abstractmethod
    def get_output(self, frame: np.ndarray) -> Tuple[
        np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        '''
        Устаревший метод
        :param frame: Изображение
        :return: Четыре numpy массива
        '''
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> YOLOBaseOut | YOLOPoseOut:
        pass


class YOLOAdapter(AdapterMixin):
    def __init__(self, model_path='yolo26n.pt', track: bool = True, 
                 image_size: int | Tuple[int, int] = 640, 
                 classes: list[int] | None = None,
                 conf=0.25, iou=0.45, device='cuda'):
        """
        Базовая yolo bbox.
        :param model_path: Имя/путь весов.
        :param track: Отслеживать ли объекты.
        :param image_size: Размер изображения для пересчёта координат [height, width].
        :param classes: Какие классы искать.
        :param conf: Порог точности.
        :param iou: Порог iou.
        :param device: Устройство для распознавания.
        """
        self.model = YOLO(model_path)
        self.track = track
        self.image_size = image_size
        self.classes = classes
        self.conf_thres = conf
        self.iou_thres = iou
        self.device = device

        self._model_kwargs = {}
        self.update_kwargs()

    def update_kwargs(self, **kwargs):
        self._model_kwargs = {
                                 "imgsz": self.image_size,
                                 "classes": self.classes,
                                 "conf": self.conf_thres,
                                 "iou": self.iou_thres,
                                 "device": self.device,
                                 "verbose": False,
                             } | kwargs

    def _run_inference(self, frame):
        if not self.track:
            return self.model.predict(frame, **self._model_kwargs)
        else:
            return self.model.track(
                frame,
                persist=True,
                **self._model_kwargs
            )
        
    def infer(self, frame: np.ndarray):
        results = self._run_inference(frame=frame)
        res = results[0].boxes.data.cpu().numpy() if len(results) != 0 else None
        return YOLOBaseOut(frame.shape[:2], res)

    def get_output(self, frame):
        dto_bb: YOLOBaseOut = self.infer(frame)
        return dto_bb.old_format


    
