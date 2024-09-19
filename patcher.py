import numpy as np
import depthai as dai
from util.functions import nms_boxes, xywh2xyxy
from tiling import Tiling
import torch

class Patcher(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Patcher"
        self.tile_manager = None
        self.conf_thresh = 0.3
        self.iou_thresh = 0.4

        self.tile_buffer = []
        self.current_timestamp = None
        self.expected_tiles_count = 0

    def set_conf_thresh(self, conf_thresh: float) -> None:
        self.conf_thresh = conf_thresh

    def set_iou_thresh(self, iou_thresh: float) -> None:
        self.iou_thresh = iou_thresh

    def build(self, tile_manager: Tiling, nn: dai.Node.Output):
        self.tile_manager = tile_manager
        if self.tile_manager.x is None or self.tile_manager.grid_size is None or self.tile_manager.overlap is None:
            raise ValueError("Tile dimensions, grid size, or overlap not initialized.")
        self.expected_tiles_count = self.tile_manager.grid_size[0] * self.tile_manager.grid_size[1]
        self.sendProcessingToPipeline(True)
        self.link_args(nn)
        return self

    def process(self, nn_output: dai.NNData) -> None:
        timestamp = nn_output.getTimestamp()
        device_timestamp = nn_output.getTimestampDevice()

        if self.current_timestamp is None:
            self.current_timestamp = timestamp

        if self.current_timestamp != timestamp and len(self.tile_buffer) > 0:
            self._send_output(self.current_timestamp, device_timestamp)
            self.tile_buffer = []

        self.current_timestamp = timestamp
        tile_index = nn_output.getSequenceNum()

        bboxes = self._extract_bboxes(nn_output)
        mapped_bboxes = self._map_bboxes_to_tile(bboxes, tile_index)
        self.tile_buffer.append(mapped_bboxes)

        if len(self.tile_buffer) == self.expected_tiles_count:
            self._send_output(timestamp, device_timestamp)
            self.tile_buffer = []

    def _extract_bboxes(self, nn_output: dai.NNData):
        """
        Extract bounding boxes and other necessary information from the neural network output.
        """
        output_tensor = nn_output.getTensor("output").astype(np.float16).reshape(10647, -1)
        output_tensor = np.expand_dims(output_tensor, axis=0)

        prediction = torch.from_numpy(output_tensor)
        if prediction.dtype is torch.float16:
            prediction = prediction.float()

        conf_mask = prediction[..., 4] > self.conf_thresh
        prediction = prediction[conf_mask]
        boxes = xywh2xyxy(prediction[:, :4])
        bboxes = torch.cat((boxes, prediction[:, 4:]), 1).numpy()
        
        return bboxes

    def _map_bboxes_to_tile(self, bboxes, tile_index):
        tile_x, tile_y = self._get_tile_coordinates(tile_index)
        mapped_bboxes = self._adjust_bboxes_to_tile(bboxes, tile_x, tile_y)
        return mapped_bboxes

    def _get_tile_coordinates(self, tile_index):
        """
        Given a tile index, calculate the true (x, y) coordinates of the top-left corner.
        """
        if self.tile_manager is None or self.tile_manager.x is None or self.tile_manager.grid_size is None or self.tile_manager.overlap is None:
            raise ValueError("Tile dimensions, grid size, or overlap not initialized.")
        if self.tile_manager.x is None or self.tile_manager.grid_size is None or self.tile_manager.overlap is None:
            raise ValueError("Tile dimensions, grid size, or overlap not initialized.")

        tile_width, tile_height = self.tile_manager.x
        n, _ = self.tile_manager.grid_size

        row = tile_index // n
        col = tile_index % n

        x = col * tile_width * (1 - self.tile_manager.overlap)
        y = row * tile_height * (1 - self.tile_manager.overlap)

        return int(x), int(y)

    def _adjust_bboxes_to_tile(self, bboxes, tile_x, tile_y):
        """
        Adjust bounding boxes to the global image coordinates using the tile's top-left corner.
        """
        if bboxes is None or bboxes.ndim == 0 or len(bboxes) == 0:
            return []

        if self.tile_manager is None or self.tile_manager.scale is None or self.tile_manager.scaled_x is None or self.tile_manager.nn_shape is None:
            raise ValueError("Tile manager not initialized.")

        scale = self.tile_manager.scale
        scaled_width, scaled_height = self.tile_manager.scaled_x
        nn_shape = self.tile_manager.nn_shape
        x_offset = (nn_shape - scaled_width) // 2
        y_offset = (nn_shape - scaled_height) // 2
        
        adjusted_bboxes = np.empty_like(bboxes)
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4]
            x1 = (x1 - x_offset) / scale + tile_x
            y1 = (y1 - y_offset) / scale + tile_y
            x2 = (x2 - x_offset) / scale + tile_x
            y2 = (y2 - y_offset) / scale + tile_y

            if x2 <= x1 or y2 <= y1:
                continue 

            adjusted_bboxes[i, 0] = x1
            adjusted_bboxes[i, 1] = y1
            adjusted_bboxes[i, 2] = x2
            adjusted_bboxes[i, 3] = y2
            adjusted_bboxes[i, 4:] = box[4:]

        return adjusted_bboxes

    def _send_output(self, timestamp, device_timestamp):
        """
        Send the final combined bounding boxes as output when all tiles for a frame are processed.
        """
        combined_bboxes = []
        for bboxes in self.tile_buffer:
            combined_bboxes.extend(bboxes)

        if combined_bboxes:
            data_array = np.array(combined_bboxes, dtype=np.float32)
            final_bboxes = nms_boxes(data_array, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh)
            
            if len(final_bboxes) > 0:
                data_array = np.array(final_bboxes, dtype=np.float32)
            else:
                data_array = np.array([], dtype=np.float32)
        else:
            data_array = np.array([], dtype=np.float32)

        serialized_data = data_array.tobytes()
        uint8_data_list = list(np.frombuffer(serialized_data, dtype=np.uint8))
        output_buffer = dai.Buffer()
        output_buffer.setData(uint8_data_list)
        output_buffer.setTimestamp(timestamp)
        output_buffer.setTimestampDevice(device_timestamp)

        # Send the output
        self.out.send(output_buffer)
