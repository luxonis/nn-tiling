# Tride: Tiling Neural Network Pipeline with DepthAI

This project splits an input frame into smaller tiles, processes each tile with a neural network, and merges the results back into a full image with detections.

## Features
- **Image Tiling** with adjustable overlap and grid size.
- **Neural Network Integration** for efficient tile-based inference.
- **Bounding Box Merging** to maintain original coordinates.

## How to Use
1. **Run the Main Program**: 
   - To start the pipeline, run the `main.py` script.
   - Press `q` to quit the program.
   - Press `g` to toggle the grid showing on and off.

```bash
python main.py
```

## Custom Tiling with Grid Matrix

You can define a custom tiling pattern using a grid matrix, allowing you to merge adjacent tiles with the same index into larger tiles. This provides flexibility in partitioning the image and optimizes performance and detection accuracy for specific use cases.

### Defining the Grid Matrix

The grid matrix is a 2D list where each element represents a tile. Adjacent tiles with the same integer are merged into a single larger tile. Here are some examples:

```python
grid_size = (4,3) # 4 tiles horizontally and 3 tiles vertically
grid_matrix_no_merge = [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
] # the matrix has to have the shape defines in grid_size
grid_matrix_1_merge = [
    [0, 1, 0, 1],
    [2, 2, 1, 0],
    [2, 2, 0, 1],
] # the matrix has to have the shape defines in grid_size
```
> Note: only **adjacent** same integers are recorgnized to be a tile. It is not based on the index itself.

## Choosing Grid Size and Overlap
> It is a trade-off game. Choose based on your specific usecase and just test it out few times ;).

| Parameter           | Advantages                                                                                             | Disadvantages                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Less Grid** | Fewer tiles to process. | "Far away" objects might not get detected due to reduction in dimension for the nn. Overall detail of frame is reduced. |
| **More Grid**| Better detection accuracy. Allows more detailed object detection. | More computational load. If too large, big objects will be split up into more tiles and might not be detected. |
| **Higher Overlap**   | Ensures objects at tile boundaries are not cut off. Improves detection of objects that might otherwise be split. | More computational load. |
| **Lower Overlap**    | Reduces computational load. | Objects near tile edges might be split or missed altogether. |

## Handling Overlapping Detections

> Why Overlap Happens?
> When you split the input image into tiles with overlap (to ensure no objects are cut off), the same object may appear in multiple tiles. This results in duplicate detections for the same object in overlapping regions.

We use *Non-Maximum Suppression (NMS)* to filter out overlapping detections. NMS works by:

1. Comparing Bounding Boxes
2. Computing Intersection-over-Union (IoU): It calculates the IoU between overlapping boxes. IoU is the ratio of the overlapping area to the total area covered by both bounding boxes.
3. Filtering Boxes: If the IoU exceeds a certain threshold (e.g., 0.5), NMS keeps the bounding box with the highest confidence score and discards the others.

## Handling Split-up Objects

> Why Split-up Objects Happen?
> When large objects span multiple tiles, they can be split into partial detections in each tile. These partial detections need to be merged into one detection for the object.

**TO BE IMPLEMENTED**

Idea for Handling Split-up Objects:
Detect in the Full Frame: Instead of detecting objects in small tiles only, process a larger frame or full image as well.
