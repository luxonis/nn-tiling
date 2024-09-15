import depthai as dai
import numpy as np
from util.functions import to_planar, non_max_suppression
from util.nympy_buffer import NumpyBuffer

class Tiling(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = dai.Node.Output(self)
        self.tiles_output = dai.Node.Output(self)
        self.nn_input = None
        self.overlap = None
        self.nn_shape = None
        self.nn_output_queue = None
        self.nn_path = None
        self.conf_thresh = 0.3
        self.iou_thresh = 0.4
        self.tiles = None

    def set_conf_thresh(self, conf_thresh: float) -> None:
        self.conf_thresh = conf_thresh

    def set_iou_thresh(self, iou_thresh: float) -> None:
        self.iou_thresh = iou_thresh

    def set_nn_output_queue(self, nn_output_q: dai.MessageQueue) -> None:
        """
        Sets the output queue of the neural network node, from which the inference results are retrieved.
        """
        self.nn_output_queue = nn_output_q

    def build(self, overlap: float, img_output: dai.Node.Output, nn_input: dai.InputQueue, nn_shape: tuple, img_shape: tuple, nn_path: str) -> "Tiling":
        self.sendProcessingToPipeline(True)
        self.link_args(img_output)
        self.nn_input = nn_input
        self.overlap = overlap
        self.nn_shape = nn_shape
        self.nn_path = nn_path
        self.tiles = self.calculate_tiles(img_shape)
        return self

    def process(self, img_frame) -> None:
        frame: np.ndarray = img_frame.getCvFrame()
        img_height, img_width, _ = frame.shape

        tiles = self.calculate_tiles((img_height, img_width))
        tiles_buffer = NumpyBuffer(np.array(tiles), img_frame)
        self.tiles_output.send(tiles_buffer)
        
        boxes = []

        for (x1, y1, x2, y2) in tiles:
            tile = frame[y1:y2, x1:x2]

            tile_img_frame = self._create_img_frame(tile, img_frame.getTimestamp())
            assert self.nn_input is not None
            self.nn_input.send(tile_img_frame)

            assert self.nn_output_queue is not None
            nn_output_data: dai.NNData = self.nn_output_queue.get() # slow

            if nn_output_data is not None:
                # if nn_output_data.getTimestamp() != tile_img_frame.getTimestamp():
                #     print("Mismatched timestamp, discarding output")
                #     continue  # Skip the current output if it doesn't match the tile's timestamp

                # Process the output
                tiled_boxes = self._process_nn_output(nn_output_data, (x1, y1, x2, y2))
                boxes.extend(tiled_boxes)

        output_buffer = NumpyBuffer(np.array(boxes), img_frame)
        self.output.send(output_buffer)

    def _process_nn_output(self, nn_output: dai.NNData, tile_coords) -> list:
        """
        Extracts the bounding boxes from the NN output and maps them back to the original image coordinates.
        """
        output_tensor = nn_output.getTensor("output").astype(np.float16).reshape(10647, -1)
        output_tensor = np.expand_dims(output_tensor, axis=0)
        
        boxes = non_max_suppression(output_tensor, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh)
        boxes = np.array(boxes[0])

        if boxes is None or boxes.ndim == 0:
            return []

        # Translate box coordinates back to original image size
        x1_tile, y1_tile, _, _ = tile_coords

        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]

            # Scale the coordinates based on the tile size and adjust to global image coordinates
            assert self.nn_shape is not None
            x1 = int(x1) + x1_tile
            y1 = int(y1) + y1_tile
            x2 = int(x2) + x1_tile
            y2 = int(y2) + y1_tile

            scaled_boxes.append([x1, y1, x2, y2, box[4], box[5]])  # (x1, y1, x2, y2, conf, class)

        return scaled_boxes

    def _create_img_frame(self, tile: np.ndarray, timestamp) -> dai.ImgFrame:
        """
        Creates an ImgFrame from the tile, which is then sent to the neural network input queue.
        """
        img_frame = dai.ImgFrame()
        assert(self.nn_shape is not None)
        img_frame.setData(to_planar(tile, (self.nn_shape[0], self.nn_shape[1])))
        img_frame.setWidth(self.nn_shape[1])
        img_frame.setHeight(self.nn_shape[0])
        img_frame.setType(dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(timestamp)
        return img_frame

    def calculate_tiles(self, image_shape):
        """
        Calculates the number of tiles needed to cover the image with the specified overlap.

        image_shape: tuple (height, width) of the input image
        nn_shape: tuple (height, width) of the neural network input size (e.g., (416, 416))
        overlap: percentage overlap between tiles (e.g., 0.2 means 20% overlap)

        Returns a list of (x1, y1, x2, y2) for each tile.
        """
        assert self.nn_shape is not None
        img_height, img_width = image_shape
        nn_height, nn_width = self.nn_shape
        
        # # Calculate the top-left corner of the tile such that it is centered
        # x1 = (img_width - nn_width) // 2
        # y1 = (img_height - nn_height) // 2
        # x2 = x1 + nn_width
        # y2 = y1 + nn_height
        #
        # # Return the one middle tile
        # return [(x1, y1, x2, y2)]

        # Step size with overlap
        assert self.overlap is not None
        step_x = int(nn_width * (1 - self.overlap))
        step_y = int(nn_height * (1 - self.overlap))

        tiles = []

        for y in range(0, img_height, step_y):
            y2 = y + nn_height
            for x in range(0, img_width, step_x):
                x2 = x + nn_width

                # If the tile extends out of the frame, adjust it
                if x2 > img_width:
                    x = img_width - nn_width  # Backup to fit
                    x2 = img_width
                if y2 > img_height:
                    y = img_height - nn_height  # Backup to fit
                    y2 = img_height

                tiles.append((x, y, x2, y2))

                # Break out early if the tile exceeds the frame size
                if x2 >= img_width:
                    break
            if y2 >= img_height:
                break

        return tiles
