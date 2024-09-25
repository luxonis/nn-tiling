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
        self.expected_tiles_count = len(self.tile_manager.tile_positions) 
        self.sendProcessingToPipeline(True)
        self.link_args(nn)
        return self

    def process(self, nn_output: dai.NNData) -> None:
        timestamp = nn_output.getTimestamp()
        device_timestamp = nn_output.getTimestampDevice()

        if self.current_timestamp is None:
            self.current_timestamp = timestamp

        if self.current_timestamp != timestamp and len(self.tile_buffer) > 0:
            # new frame started, send the output for the previous frame
            self._send_output(self.current_timestamp, device_timestamp)
            self.tile_buffer = []

        self.current_timestamp = timestamp
        tile_index = nn_output.getSequenceNum()

        bboxes = self._extract_bboxes(nn_output)
        mapped_bboxes = self._map_bboxes_to_tile(bboxes, tile_index)
        self.tile_buffer.append(mapped_bboxes)

        # print info
        # print("Tile index:", tile_index)
        # print("Bounding boxes:", len(mapped_bboxes))
        # print("Tile buffer length:", len(self.tile_buffer))
        # print()

        if len(self.tile_buffer) == self.expected_tiles_count:
            # all tiles processed
            # print("------------------------------------------------")
            # print("All tiles processed.")
            # print("------------------------------------------------")
            # print()
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
        tile_info = self._get_tile_info(tile_index)
        if tile_info is None:
            return []
        tile_x, tile_y = tile_info['coords'][:2]
        scaled_width, scaled_height = tile_info['scaled_size']
        tile_scale = min(self.tile_manager.nn_shape / (tile_info['coords'][2] - tile_info['coords'][0]),
                         self.tile_manager.nn_shape / (tile_info['coords'][3] - tile_info['coords'][1]))
        mapped_bboxes = self._adjust_bboxes_to_tile(bboxes, tile_x, tile_y, scaled_width, scaled_height, tile_scale)
        return mapped_bboxes

    def _get_tile_info(self, tile_index):
        """
        Retrieves the tile's coordinates and scaled dimensions based on the tile index.
        """
        if self.tile_manager is None or self.tile_manager.tile_positions is None:
            raise ValueError("Tile manager or tile positions not initialized.")
        if tile_index >= len(self.tile_manager.tile_positions):
            return None
        return self.tile_manager.tile_positions[tile_index]

    def _adjust_bboxes_to_tile(self, bboxes, tile_x, tile_y, scaled_width, scaled_height, tile_scale):
        """
        Adjust bounding boxes to the global image coordinates using the tile's top-left corner and scaling.
        """
        if bboxes is None or bboxes.ndim == 0 or len(bboxes) == 0:
            return []

        nn_shape = self.tile_manager.nn_shape
        x_offset = (nn_shape - scaled_width) // 2
        y_offset = (nn_shape - scaled_height) // 2

        adjusted_bboxes = []
        for box in bboxes:
            x1, y1, x2, y2 = box[:4]

            # Reverse scaling and offset applied during tiling
            x1 = (x1 - x_offset) / tile_scale + tile_x
            y1 = (y1 - y_offset) / tile_scale + tile_y
            x2 = (x2 - x_offset) / tile_scale + tile_x
            y2 = (y2 - y_offset) / tile_scale + tile_y

            if x2 <= x1 or y2 <= y1:
                continue

            adjusted_box = [x1, y1, x2, y2] + box[4:].tolist()
            adjusted_bboxes.append(adjusted_box)

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
