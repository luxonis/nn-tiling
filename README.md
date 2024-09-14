# Tiling Neural Network Pipeline with DepthAI

This project splits an input frame into smaller tiles, processes each tile with a neural network, and merges the results back into a full image with detections.

## Features
- **Image Tiling** with adjustable overlap.
- **Neural Network Integration** for efficient tile-based inference.
- **Bounding Box Merging** to maintain original coordinates.

## To-Do List

### 1. Handle Overlapped Detections
- **Problem**: When using overlapping tiles, the same object may be detected in multiple tiles, resulting in duplicate detections.
- **Solution**: Implement logic to merge or filter out overlapping detections. This can be done by:
  - Comparing the bounding boxes of detections from adjacent tiles.
  - Using Non-Maximum Suppression (NMS) to merge overlapping boxes based on Intersection-over-Union (IoU) threshold.
  - Keeping the detection with the highest confidence score and removing duplicates.

### 2. Handle Split-up Objects
- **Problem**: Large objects that span multiple tiles may be split up, resulting in multiple partial detections that need to be combined.
- **Solution**: Implement a strategy to combine split detections across tiles:
  - Identify bounding boxes in adjacent tiles that could belong to the same object.
  - Merge or stitch together these detections to form a single bounding box for the object.
  - Consider the object's class and confidence score to assist in determining which detections to merge.

### Next Steps:
- Develop and test algorithms for overlapping detection handling and object stitching across tiles.
- Tune Non-Maximum Suppression (NMS) thresholds for both overlap filtering and merging split-up objects.
- Validate against various test cases with different grid sizes and overlap percentages.
