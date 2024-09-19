import depthai as dai
import cv2
import numpy as np
import time

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
        self.draw_grid_bool = True

        self.fps = None
        self.frame_count = 0 
        self.start_time = time.time()

        self.font_scale = 1 
        self.font_thickness = 2 
        self.padding = 8

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

    def process(self, img_frame, boxes_buffer: dai.Buffer) -> None:
        frame = img_frame.getCvFrame()
        if self.draw_grid_bool: self.draw_grid(frame)
        self._draw_fps(frame)

        received_data = boxes_buffer.getData()
        boxes_data = np.array(received_data, dtype=np.uint8).view(np.float32).reshape(-1, 6) # x1, y1, x2, y2, conf, cls

        # Draw bounding boxes
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls = box
            if self.label_map is None:
                raise ValueError("Label map is not initialized.")
            label = f"{self.label_map[int(cls)]}: {conf:.2f}"

            color = (255, 87, 51) 
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
            
            cv2.rectangle(frame, 
                  (int(x1), int(y1) - label_height - baseline - self.padding),
                  (int(x1) + label_width + self.padding, int(y1)), 
                  color, -1)

            cv2.putText(frame, label, (int(x1) + self.padding // 2, int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.font_scale, (255, 255, 255), self.font_thickness)

        cv2.imshow(self.name, frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stopPipeline()
        elif key == ord('g'):
            self.draw_grid_bool = not self.draw_grid_bool

    def _draw_fps(self, frame) -> None:
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time

        if self.fps is None:
            fps_text = "FPS: Calculating..."
        else:
            fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (0, 0, 0), 20, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (255, 255, 255), 3, cv2.LINE_AA)
        
    def draw_grid(self, frame: np.ndarray) -> None:
        if self.x is None or self.grid_size is None or self.overlap is None:
            print("Error: Tile dimensions, grid size, or overlap are not set.")
            return

        tile_width, tile_height = self.x
        n_tiles_w, n_tiles_h = self.grid_size
        img_height, img_width, _ = frame.shape

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
                
                self._draw_filled_rect_with_alpha(frame, (x1, y1), (x2, y2), color)
                
        # draw grid info
        grid_size_text = f"{n_tiles_h}x{n_tiles_w} overlap: {self.overlap}" 
        text_x = img_width // 2
        text_y = img_height - 30 

        cv2.putText(frame, grid_size_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def _draw_filled_rect_with_alpha(self, frame, top_left, bottom_right, color_with_alpha):
        overlay = frame.copy()
        output = frame.copy()
        color = color_with_alpha[:3] 
        alpha = color_with_alpha[3]
        
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        np.copyto(frame, output)

