# functions taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py

import torch
from torch import from_numpy
import torchvision
import time
import numpy as np
import cv2

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Converts the input image `arr` (NumPy array) to the planar format expected by depthai.
    The image is resized to the dimensions specified in `shape`.
    
    Parameters:
    - arr: Input NumPy array (image).
    - shape: Target dimensions (width, height).
    
    Returns:
    - A 1D NumPy array with the planar image data.
    """
    if arr.shape[:2] == shape:
        resized = arr 
    else:
        resized = cv2.resize(arr, shape)

    return resized.transpose(2, 0, 1).flatten()

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    prediction=torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        #x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def nms_boxes(boxes, conf_thresh=0.3, iou_thresh=0.4):
    """
    Applies Non-Maximum Suppression (NMS) on bounding boxes.

    Parameters:
    - boxes: NumPy array of shape (num_boxes, 5 + num_classes).
    - conf_thresh: Confidence threshold for filtering boxes.
    - iou_thresh: IoU threshold for NMS.

    Returns:
    - A NumPy array of bounding boxes after NMS.
    """
    if len(boxes) == 0:
        return np.array([])

    num_classes = boxes.shape[1] - 5
    obj_conf = boxes[:, 4]
    prediction = boxes[obj_conf >= conf_thresh]
    if len(prediction) == 0:
        return np.array([])

    prediction[:, 5:] *= prediction[:, 4:5] # conf = obj_conf * cls_conf

    final_boxes = []

    for class_idx in range(num_classes):
        class_scores = prediction[:, 5 + class_idx]
        class_mask = class_scores >= conf_thresh
        class_boxes = prediction[class_mask]
        if len(class_boxes) == 0:
            continue

        x1 = class_boxes[:, 0]
        y1 = class_boxes[:, 1]
        x2 = class_boxes[:, 2]
        y2 = class_boxes[:, 3]
        
        scores = class_scores[class_mask]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1] # descending 

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU of the kept box with the rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        # Append kept boxes and assign class index
        class_boxes = class_boxes[keep]
        class_boxes = np.hstack((class_boxes[:, :5], np.full((len(keep), 1), class_idx, dtype=np.int32)))
        final_boxes.append(class_boxes)

    if len(final_boxes) == 0:
        return np.array([])

    return np.vstack(final_boxes)
