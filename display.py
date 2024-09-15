import depthai as dai
import cv2
import numpy as np

class Display(dai.node.HostNode):
    _wait_key_instance = None

    def __init__(self) -> None:
        super().__init__()
        self.name = "Display"
        self.process_wait_key = False
        self.label_map = None
        self.x = None
        self.overlap = None
        self.grid_size = None

        if Display._wait_key_instance is None:
            self.process_wait_key = True
            Display._wait_key_instance = self

    def build(self, frame, boxes, x, overlap, grid_size, label_map):
        self.sendProcessingToPipeline(True)
        self.link_args(frame, boxes)
        self.label_map = label_map
        self.x = x
        self.overlap = overlap
        self.grid_size = grid_size
        return self

    def setName(self, name: str) -> None:
        self.name = name

    def setWaitForExit(self, wait: bool) -> None:
        if Display._wait_key_instance is None and wait:
            self.process_wait_key = True
            Display._wait_key_instance = self
        elif Display._wait_key_instance is self and not wait:
            self.process_wait_key = False
            Display._wait_key_instance = None

    def draw_grid(self, frame: np.ndarray) -> None:
        if self.x is None or self.grid_size is None or self.overlap is None:
            print("Error: Tile dimensions, grid size, or overlap are not set.")
            return

        tile_width, tile_height = self.x
        n_tiles_w, n_tiles_h = self.grid_size
        img_height, img_width, _ = frame.shape

        # Generate a list of random colors with alpha transparency
        np.random.seed(432) 
        colors = [
            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), 0.3) 
            for _ in range(n_tiles_w * n_tiles_h)
        ]
        
        color_idx = 0
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                x1 = int(j * tile_width * (1 - self.overlap))
                y1 = int(i * tile_height * (1 - self.overlap))

                x2 = min(int(x1 + tile_width), img_width)
                y2 = min(int(y1 + tile_height), img_height)

                color = colors[color_idx]
                color_idx += 1
                
                # Draw the rectangle for the current tile with alpha blending
                self._draw_filled_rect_with_alpha(frame, (x1, y1), (x2, y2), color)

    def _draw_filled_rect_with_alpha(self, frame, top_left, bottom_right, color_with_alpha):
        overlay = frame.copy()
        output = frame.copy()

        color = color_with_alpha[:3]  # Extract BGR color
        alpha = color_with_alpha[3]   # Extract alpha

        # Draw a filled rectangle on the overlay
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Copy the output back to the original image
        np.copyto(frame, output)

    def process(self, img_frame, boxes: dai.Buffer) -> None:
        frame = img_frame.getCvFrame()
        self.draw_grid(frame)
        
        boxes_data = np.array(boxes.getData()).reshape(-1, 6)  # Expecting (x1, y1, x2, y2, conf, cls)

        # Draw bounding boxes
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls = box
            assert self.label_map is not None
            label = f"{self.label_map[int(cls)]}: {conf:.2f}"

            color = (0, 255, 0)
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(frame, label,(int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow(self.name, frame)

        if self.process_wait_key and cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
