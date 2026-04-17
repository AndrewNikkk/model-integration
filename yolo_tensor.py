import numpy as np


class BboxAccsessor:
    '''Аксессор для доступа к bounding boxes'''

    __slots__=('_parent')

    def __init__(self, parent):
        self._parent = parent

    @property
    def xywh(self):
        x = self._parent.data[:, :4]
        assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
        y = np.empty_like(x, dtype=x.dtype)
        x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        y[..., 0] = (x1 + x2) / 2
        y[..., 1] = (y1 + y2) / 2
        y[..., 2] = x2 - x1
        y[..., 3] = y2 - y1
        return y
    
    @property
    def xywhn(self):
        xywh = self.xywh
        xywh[..., [0, 2]] /= self._parent.img_size[1]
        xywh[..., [1, 3]] /= self._parent.img_size[0]
        return xywh


    @property
    def xyxy(self):
        return self._parent.data[:, :4]
    
    @property
    def xyxyn(self):
        if self._parent.img_size is None:
            raise ValueError("orig_shape is required for normalized coordinates")
        h, w = int(self._parent.img_size[0]), int(self._parent.img_size[1])
        if h <= 0 or w <= 0:
            raise ValueError("orig_shape must have positive height and width")
        out = self._parent.data[:, :4].copy().astype(np.float32, copy=False)
        out[..., [0, 2]] /= w
        out[..., [1, 3]] /= h
        return out
    

class KeypointsAccsessor:
    '''Аксессор для доступа к keypints'''
    
    __slots__ = ('_parent')

    def __init__(self, parent):
        self._parent = parent

    @property
    def xy(self):
        return self._parent._keypoints_data[..., :2]

    @property
    def xyn(self):
        if self._parent.img_size is None:
            raise ValueError("orig_shape is required for normalized coordinates")
        h, w = int(self._parent.img_size[0]), int(self._parent.img_size[1])
        xy = self.xy.copy()
        xy[..., 0] /= w
        xy[..., 1] /= h
        return xy

    @property
    def conf(self):
        return self._parent._keypoints_data[..., 2]


class MasksAccsessor:
    '''Аксессор для доступа к masks'''

    __slots__ = ('parent')
    def __init__(self, parent):
        self.parent = parent

    @property
    def xy(self):
        return self.parent._masks_data[..., :2]

class YOLOBaseOut:
    
    def __init__(self, data, orig_shape):
        if data is None:
            self.data = np.zeros((0, 7), dtype=np.float32)
        else:
            self.data = data
        self.img_size = orig_shape
        self.bboxes = BboxAccsessor(self)


    @classmethod
    def from_columns(cls, xyxy, ids, confs, classes, orig_shape):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        ids = np.asarray(ids, dtype=np.float32).ravel()
        confs = np.asarray(confs, dtype=np.float32).ravel()
        classes = np.asarray(classes, dtype=np.float32).ravel()
        n = xyxy.shape[0]
        if len(ids) != n or len(confs) != n or len(classes) != n:
            raise ValueError("Lengths of xyxy/ids/confs/classes must match")
        data = np.empty((n, 7), dtype=np.float32)
        data[:, :4] = xyxy
        data[:, 4] = ids
        data[:, 5] = confs
        data[:, 6] = classes
        return cls(data, orig_shape)


    @property
    def ids(self):
        return self.data[:, 4].astype(int) 
    
    
    @property
    def confs(self):
        return self.data[:, 5]
    

    @property
    def classes(self):
        return self.data[:, 6].astype(int)
    

class YOLOPoseOut(YOLOBaseOut):
    def __init__(self, data, orig_shape, keypoints_data=None, num_keypoints=17):
        super().__init__(data, orig_shape)
        if keypoints_data is None:
            n = len(self.data)
            self._keypoints_data = np.zeros((n, num_keypoints, 3), dtype=np.float32)
            self.num_keypoints = num_keypoints
        else:
            kp = np.asarray(keypoints_data, dtype=np.float32)
            if kp.ndim != 3 or kp.shape[0] != len(self.data) or kp.shape[2] != 3:
                raise ValueError(
                    f"keypoints_data must be (N,K,3), got {kp.shape}, N={len(self.data)}"
                )
            self._keypoints_data = kp    
            self.num_keypoints = self._keypoints_data.shape[1]
        self.keypoints = KeypointsAccsessor(self)


    @classmethod
    def from_columns(cls, xyxy, ids, confs, classes, keypoints, orig_shape):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        ids = np.asarray(ids, dtype=np.float32).ravel()
        confs = np.asarray(confs, dtype=np.float32).ravel()
        classes = np.asarray(classes, dtype=np.float32).ravel()

        n = xyxy.shape[0]
        if len(ids) != n or len(confs) != n or len(classes) != n:
            raise ValueError("Lengths of xyxy/ids/confs/classes must match")

        data = np.empty((n, 7), dtype=np.float32)
        data[:, :4] = xyxy
        data[:, 4] = ids
        data[:, 5] = confs
        data[:, 6] = classes

        if keypoints is None:
            return cls(data=data, orig_shape=orig_shape, keypoints_data=None)
        else:
            kp = np.asarray(keypoints, dtype=np.float32)

            if kp.ndim != 3 or kp.shape[0] != n or kp.shape[2] != 3:
                raise ValueError(f"keypoints must be (N,K,3), got {kp.shape}")

            return cls(data=data, orig_shape=orig_shape, keypoints_data=kp)


class YOLOSegOut(YOLOBaseOut):
    def __init__(self, data, orig_shape, masks=None):
        super().__init__(data, orig_shape)
        if masks is None:
            n = len(self.data)
            self._masks_data = np.zeros((n, orig_shape[0], orig_shape[1]), dtype=np.float32)
        else:
            masks = np.asarray(masks, dtype=np.float32)
            if masks.ndim != 3 or masks.shape[0] != len(self.data):
                raise ValueError(f"masks must be (N, orig_shape[0], orig_shape[1]), got {masks.shape}, N={len(self.data)}")
            self._masks_data = masks

    @classmethod
    def from_columns(cls, xyxy, ids, confs, classes, masks, orig_shape):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        ids = np.asarray(ids, dtype=np.float32).ravel()
        confs = np.asarray(confs, dtype=np.float32).ravel()
        classes = np.asarray(classes, dtype=np.float32).ravel()
        masks = np.asarray(masks, dtype=np.float32).ravel()

        n = xyxy.shape[0]
        if len(ids) != n or len(confs) != n or len(classes) != n:
            raise ValueError("Lengths of xyxy/ids/confs/classes must match")

        data = np.empty((n, 7), dtype=np.float32)
        data[:, :4] = xyxy
        data[:, 4] = ids
        data[:, 5] = confs
        data[:, 6] = classes
        return cls(data=data, orig_shape=orig_shape)

    @property
    def ids(self):
        return self.data[:, 4].astype(int)
    
    @property
    def confs(self):
        return self.data[:, 5]