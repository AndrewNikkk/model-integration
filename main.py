import numpy as np
import cv2
from data_formats.yolo import YOLOBaseOut
from yolo26 import YOLOAdapter
from yolo26_pose import YOLOPoseAdapter
# from seg_adapter import YOLOSegAdapter
# from obb_adapter import YOLOOBBAdapter

if __name__ == "__main__":
    image = cv2.imread('pose_test.jpg')

    yolo_adapter = YOLOAdapter()
    

    yolo_adapter.set_mode('track')

    print("=============================ТЕНЗОР ВЫВОД====================================")

    res = yolo_adapter.get_output_tensor(image)

    print(f'BBOXES.XY: {res.bboxes.xyxy}')
    print(f'BBOXES.XYXYN: {res.bboxes.xyxyn}')
    print(f'CONFS: {res.confs}')
    print(f'IDS: {res.ids}')
    print(f'CLASSES: {res.classes}')

    print("=============================СТАРЫЙ ВЫВОД====================================")


    bboxes, classes, ids = yolo_adapter.get_output(image)

    print(f'BBOXES: {bboxes}')
    print(f'CLASSES: {classes}')
    print(f'IDS: {ids}')


    print("===============================СОЗДАНИЕ ЧЕРЕЗ СПИСКИ==================================")

    bboxes = [2.33, 2.33, 2.33, 2.33]
    confs = [0.28]
    ids = [1]
    classes = [0]
    orig_shape = [10, 10]

    tensor = YOLOBaseOut.from_columns(xyxy=bboxes, confs=confs, ids=ids, classes=classes, orig_shape=orig_shape)


    print(f'BBOXES.XYXY: {tensor.bboxes.xyxy}')
    print(f'BBOXES.XYXYN: {tensor.bboxes.xyxyn}')
    print(f'CONFS: {tensor.confs}')
    print(f'IDS: {tensor.ids}')
    print(f'CLASSES: {tensor.classes}')



    print("===============================ТЕНЗОР ВЫВОД==================================")


    yolo_pose = YOLOPoseAdapter()

    yolo_pose.set_mode('track')

    pose_res = yolo_pose.get_output_tensor(image)


    print(f'BBOXES.XYXY: {pose_res.bboxes.xyxy}')
    print(f'CLASSES: {pose_res.classes}')
    print(f'CONFS: {pose_res.confs}')
    print(f'IDS: {pose_res.ids}')
    print(f'KEYPOINTS.XY: {pose_res.keypoints.xy}')
    print(f'KEYPOINTS.XYN: {pose_res.keypoints.xyn}')


    print("===============================СТАРЫЙ ВЫВОД==================================")


    bboxes, classes, ids = yolo_pose.get_output(image)

    print(f'BBOXES: {bboxes}')
    print(f'CLASSES: {classes}')
    print(f'IDS: {ids}')
    print(f'KEYPOINTS: {yolo_pose.keypoints_xy}')
    print(f'KEYPOINTS_CONFS: {yolo_pose.keypoints_confs}')
    