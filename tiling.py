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
        self.grid_size = None
        self.nn_output_queue = None
        self.nn_path = None
        self.conf_thresh = 0.3
        self.iou_thresh = 0.4
        self.x = None # vector [x,y] of the tile

    def set_conf_thresh(self, conf_thresh: float) -> None:
        self.conf_thresh = conf_thresh

    def set_iou_thresh(self, iou_thresh: float) -> None:
        self.iou_thresh = iou_thresh

    def set_nn_output_queue(self, nn_output_q: dai.MessageQueue) -> None:
        self.nn_output_queue = nn_output_q

    def build(self, overlap: float, img_output: dai.Node.Output, nn_input: dai.InputQueue, grid_size: tuple, img_shape: tuple, nn_path: str) -> "Tiling":
        self.sendProcessingToPipeline(True)
        self.link_args(img_output)
        self.nn_input = nn_input
        self.overlap = overlap
        self.grid_size = grid_size
        self.nn_path = nn_path
        self.x = self._calculate_tiles(grid_size, img_shape, overlap)
        return self

    def process(self, img_frame) -> None:
        self.output.send(NumpyBuffer(np.array([]), img_frame))
        return
        frame: np.ndarray = img_frame.getCvFrame()

        if self.grid_size is None or self.x is None:
            raise ValueError("Grid size or tile dimensions are not initialized.")
            
        img_height, img_width, _ = frame.shape
        tile_width, tile_height = self.x

        tiles = self._extract_tiles(frame, img_height, img_width, tile_width, tile_height)

        for tile, _ in tiles:
            tile_img_frame = self._create_img_frame(tile, img_frame.getTimestamp())
            assert self.nn_input is not None
            self.nn_input.send(tile_img_frame)

        boxes = []
        for _, (x1, y1, x2, y2) in tiles:
            assert self.nn_output_queue is not None
            try:
                nn_output_data: dai.NNData = self.nn_output_queue.get()
            except Exception as e:
                print(f"Error while getting NN output: {e}")
                continue

            if nn_output_data is not None:
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
            x1 = int(x1) + x1_tile
            y1 = int(y1) + y1_tile
            x2 = int(x2) + x1_tile
            y2 = int(y2) + y1_tile

            scaled_boxes.append([x1, y1, x2, y2, box[4], box[5]])  # (x1, y1, x2, y2, conf, class)

        return scaled_boxes

    def _create_img_frame(self, tile: np.ndarray, timestamp) -> dai.ImgFrame:
        pass
        # """
        # Creates an ImgFrame from the tile, which is then sent to the neural network input queue.
        # """
        # img_frame = dai.ImgFrame()
        # assert(self.nn_shape is not None)
        # img_frame.setData(to_planar(tile, (self.nn_shape[0], self.nn_shape[1])))
        # img_frame.setWidth(self.nn_shape[1])
        # img_frame.setHeight(self.nn_shape[0])
        # img_frame.setType(dai.ImgFrame.Type.BGR888p)
        # img_frame.setTimestamp(timestamp)
        # return img_frame

    def _calculate_tiles(self, grid_size, img_shape, overlap):
        """
        Calculate tile dimensions (x, y) given grid size, image shape, and overlap.
        """
        n_tiles_w, n_tiles_h = grid_size
        
        A = np.array([
            [n_tiles_w * (1 - overlap) + overlap, 0],
            [0, n_tiles_h * (1 - overlap) + overlap]
        ])
        
        b = np.array(img_shape)
        
        tile_dims = np.linalg.inv(A).dot(b)
        tile_width, tile_height = tile_dims
        
        return tile_width, tile_height

    def _extract_tiles(self, frame, img_height, img_width, tile_width, tile_height):
        """
        Extracts tiles from the given frame based on the grid and overlap.
        Returns a list of tiles with their coordinates.
        """
        if self.grid_size is None or self.overlap is None:
            raise ValueError("Grid size or overlap is not initialized.")
        
        tiles = []
        n_tiles_w, n_tiles_h = self.grid_size

        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate the top-left and bottom-right coordinates of the tile
                x1 = int(j * tile_width * (1 - self.overlap))
                y1 = int(i * tile_height * (1 - self.overlap))
                x2 = min(int(x1 + tile_width), img_width)
                y2 = min(int(y1 + tile_height), img_height)

                # Extract the tile from the frame
                tile = frame[y1:y2, x1:x2]
                tiles.append((tile, (x1, y1, x2, y2)))

        return tiles
