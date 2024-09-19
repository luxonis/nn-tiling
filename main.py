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
conf_thresh = 0.3
iou_thresh = 0.4 

NN_SHAPE = 416
IMG_SHAPE = (1280, 720)
# IMG_SHAPE = (3840, 2160)

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
    # replay.setReplayVideoFile(Path('videos/4k_traffic.mp4'))
    replay.setSize(IMG_SHAPE)
    replay.setFps(1)
    cam_out = replay.out

    overlap = 0.2
    grid_size = (3, 2) # (number of tiles horizontally, number of tiles vertically)
   
    tile_manager = pipeline.create(Tiling).build(
        img_output=cam_out,
        nn_input=nn_input_queue,
        img_shape=IMG_SHAPE, 
        overlap=overlap,
        grid_size=grid_size,
        nn_path=nn_path,
        nn_shape=NN_SHAPE
    )
    
    tile_manager.set_nn_output_queue(nn_output_queue)
    tile_manager.set_conf_thresh(conf_thresh)
    tile_manager.set_iou_thresh(iou_thresh)
    tile_manager.out.link(detection_nn.input)

    # manip = pipeline.create(dai.node.ImageManip) 
    # manip.initialConfig.setResizeThumbnail(NN_SHAPE, NN_SHAPE)
    # manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    # manip.inputImage.setBlocking(True)
    # tile_manager.out.link(manip.inputImage)
    # # cam_out.link(manip.inputImage)
    # manip.out.link(detection_nn.input)

    patcher = pipeline.create(Patcher).build(
        tile_manager=tile_manager,
        nn=detection_nn.out
    )
    patcher.set_conf_thresh(conf_thresh)
    patcher.set_iou_thresh(iou_thresh)
    # patcher.inputs['nn'].setMaxSize(6)

    display = pipeline.create(Display).build(
        frame=cam_out,
        boxes=patcher.out,
        x=tile_manager.x,
        overlap=overlap,
        grid_size=grid_size,
        label_map=labelMap
    )

    pipeline.run()
