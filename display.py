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
        self.tile_positions = None 
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

    def build(self, frame, boxes, tile_positions, label_map):
        self.sendProcessingToPipeline(True)
        self.link_args(frame, boxes)
        self.label_map = label_map
        self.tile_positions = tile_positions
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
        if not self.tile_positions:
            print("Error: Tile positions are not set.")
            return

        img_height, img_width, _ = frame.shape

        np.random.seed(432)
        colors = [
            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), 0.3)
            for _ in range(len(self.tile_positions))
        ]

        for idx, tile_info in enumerate(self.tile_positions):
            x1, y1, x2, y2 = tile_info['coords']
            color = colors[idx % len(colors)]
            self._draw_filled_rect_with_alpha(frame, (int(x1), int(y1)), (int(x2), int(y2)), color)

        # Optionally, display grid info
        grid_info_text = f"Tiles: {len(self.tile_positions)}"
        text_x = img_width // 2 - 100
        text_y = img_height - 30

        cv2.putText(frame, grid_info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def _draw_filled_rect_with_alpha(self, frame, top_left, bottom_right, color_with_alpha):
        overlay = frame.copy()
        output = frame.copy()
        color = color_with_alpha[:3] 
        alpha = color_with_alpha[3]
        
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        np.copyto(frame, output)

