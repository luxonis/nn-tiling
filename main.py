#!/usr/bin/env python3
import depthai as dai
from tiling import Tiling
from display import Display
from pathlib import Path

'''
YoloV5 object detector running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob is taken from ML training examples:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks

You can clone the YoloV5_training.ipynb notebook and try training the model yourself.

'''
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

nn_path = 'models/yolov5s_default_openvino_2021.4_6shave.blob'
conf_thresh = 0.3
iou_thresh = 0.4 

NN_SHAPE = 416
IMG_SHAPE = (1280, 720)

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(Path(nn_path))

    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)
    detection_nn.setNumShavesPerInferenceThread(8)
    nn_input_queue = detection_nn.input.createInputQueue(maxSize=10, blocking=False)
    nn_output_queue = detection_nn.out.createOutputQueue(maxSize=10, blocking=False)

    # cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    # cam_out = cam.requestOutput((1920, 1080), dai.ImgFrame.Type.BGR888p)
    # cam = pipeline.create(dai.node.ColorCamera)
    # cam.setPreviewSize(1920, 1080)
    # cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # cam.setInterleaved(False)
    # cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # cam.setFps(40)
    # cam_out = cam.preview

    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setLoop(False)
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
    replay.setReplayVideoFile(Path('videos/2.1.mp4'))
    replay.setSize(IMG_SHAPE)
    # replay.setFps(10)
    cam_out = replay.out

    overlap = 0.1
    grid_size = (4,3) # (number of tiles horizontally, number of tiles vertically)
   
    tiling = pipeline.create(Tiling).build(
        img_output=cam_out,
        nn_input=nn_input_queue,
        img_shape=IMG_SHAPE, 
        overlap=overlap,
        grid_size=grid_size,
        nn_path=nn_path,
    )
    
    tiling.set_nn_output_queue(nn_output_queue)
    tiling.set_conf_thresh(conf_thresh)
    tiling.set_iou_thresh(iou_thresh)

    display = pipeline.create(Display).build(
        frame=cam_out,
        boxes=tiling.output,
        x=tiling.x,
        overlap=overlap,
        grid_size=grid_size,
        label_map=labelMap
    )

    pipeline.run()
