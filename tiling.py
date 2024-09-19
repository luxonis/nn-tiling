import depthai as dai
import numpy as np
from util.functions import to_planar
import cv2

class Tiling(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Tiling"
        self.overlap = None
        self.grid_size = None
        self.nn_shape = None
        self.x = None # vector [x,y] of the tile's dimensions
        self.scale = None
        self.scaled_x = None

    def build(self, overlap: float, img_output: dai.Node.Output, grid_size: tuple, img_shape: tuple, nn_shape: int) -> "Tiling":
        self.sendProcessingToPipeline(True)
        self.link_args(img_output)
        self.overlap = overlap
        self.grid_size = grid_size
        self.nn_shape = nn_shape
        self.x = self._calculate_tiles(grid_size, img_shape, overlap)
        
        print(f"Tile dimensions: {self.x}")
        return self

    def process(self, img_frame) -> None:
        frame: np.ndarray = img_frame.getCvFrame()

        if self.grid_size is None or self.x is None or self.nn_shape is None:
            raise ValueError("Grid size or tile dimensions are not initialized.")
            
        img_height, img_width, _ = frame.shape
        tile_width, tile_height = self.x
        tiles = self._extract_tiles(frame, img_height, img_width, tile_width, tile_height)

        tile_padded = np.zeros((self.nn_shape, self.nn_shape, 3), dtype=np.uint8)

        for index, (tile, _) in enumerate(tiles):
            tile_img_frame = self._create_img_frame(tile, img_frame, index, tile_padded)
            self.out.send(tile_img_frame)            

    def _create_img_frame(self, tile: np.ndarray, frame, tile_index, tile_padded) -> dai.ImgFrame:
        """
        Creates an ImgFrame from the tile, which is then sent to the neural network input queue.
        """
        if self.nn_shape is None or self.scaled_x is None:
            raise ValueError("NN shape or scaled tile dimensions are not initialized.")
        new_width, new_height = self.scaled_x
        tile_resized = cv2.resize(tile, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        tile_padded.fill(0)

        x_offset = (self.nn_shape - new_width) // 2 
        y_offset = (self.nn_shape - new_height) // 2
        tile_padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = tile_resized

        planar_tile = to_planar(tile_padded, (self.nn_shape, self.nn_shape))

        img_frame = dai.ImgFrame()
        img_frame.setData(planar_tile)
        img_frame.setWidth(self.nn_shape)
        img_frame.setHeight(self.nn_shape)
        img_frame.setType(dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame.getTimestamp())
        img_frame.setTimestampDevice(frame.getTimestampDevice())
        img_frame.setInstanceNum(frame.getInstanceNum())
        img_frame.setSequenceNum(tile_index)

        return img_frame

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

        # scaling of each tile 
        if self.nn_shape is None:
            raise ValueError("NN shape is not initialized.")
        self.scale = min(self.nn_shape / tile_width, self.nn_shape / tile_height)
        self.scaled_x = (int(tile_width * self.scale), int(tile_height * self.scale))
        
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
                x1 = int(j * tile_width * (1 - self.overlap))
                y1 = int(i * tile_height * (1 - self.overlap))
                x2 = min(int(x1 + tile_width), img_width)
                y2 = min(int(y1 + tile_height), img_height)

                tile = frame[y1:y2, x1:x2]
                tiles.append((tile, (x1, y1, x2, y2)))

        return tiles
