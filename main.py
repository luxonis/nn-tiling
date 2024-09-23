#!/usr/bin/env python3
import depthai as dai
from tiling import Tiling
from patcher import Patcher
from display import Display
from pathlib import Path

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
conf_thresh = 0.4
iou_thresh = 0.4 

NN_SHAPE = 416
# IMG_SHAPE = (1280, 720)
# IMG_SHAPE = (3840, 2160)
IMG_SHAPE = (1920, 1080)

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    # cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    # cam_out = cam.requestOutput(IMG_SHAPE, dai.ImgFrame.Type.BGR888p)

    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setLoop(False)
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
    # replay.setReplayVideoFile(Path('videos/2.1.mp4'))
    # replay.setReplayVideoFile(Path('videos/cattle_drone.mp4'))
    replay.setReplayVideoFile(Path('videos/cattle_drone_closed_up.mp4'))
    replay.setSize(IMG_SHAPE)
    replay.setFps(2)
    cam_out = replay.out
    
    overlap = 0.2
    grid_size = (3,3) # (number of tiles horizontally, number of tiles vertically)
    grid_matrix = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
   
    tile_manager = pipeline.create(Tiling).build(
        img_output=cam_out,
        img_shape=IMG_SHAPE, 
        overlap=overlap,
        grid_size=grid_size,
        grid_matrix=grid_matrix,
        # global_detection=False,
        nn_shape=NN_SHAPE
    )

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(Path(nn_path))
    detection_nn.setNumPoolFrames(6)
    detection_nn.input.setBlocking(False)
    detection_nn.input.setMaxSize(len(tile_manager.tile_positions))
    detection_nn.setNumInferenceThreads(2)
    detection_nn.setNumShavesPerInferenceThread(8)
    tile_manager.out.link(detection_nn.input)

    patcher = pipeline.create(Patcher).build(
        tile_manager=tile_manager,
        nn=detection_nn.out
    )
    patcher.set_conf_thresh(conf_thresh)
    patcher.set_iou_thresh(iou_thresh)

    display = pipeline.create(Display).build(
        frame=cam_out,
        boxes=patcher.out,
        tile_positions=tile_manager.tile_positions,
        label_map=labelMap
    )

    pipeline.run()
