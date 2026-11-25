from types import SimpleNamespace

import cv2
import numpy as np
import torch


NAMES = dict(enumerate(('UNKNOWN', 'RED-CIRCLE', 'RED-AMBER-CIRCLE', 'AMBER-CIRCLE', 'GREEN-CIRCLE',
                       'RED-ARROW-LEFT', 'RED-ARROW-RIGHT', 'RED-ARROW-STRAIGHT', 'AMBER-ARROW-LEFT',
                       'AMBER-ARROW-RIGHT', 'AMBER-ARROW-STRAIGHT', 'GREEN-ARROW-LEFT',
                       'GREEN-ARROW-RIGHT', 'GREEN-ARROW-STRAIGHT')))


def decode_yolox(output, size=640):
    grids, strides = [], []
    for stride in (8, 16, 32):
        grid_x, grid_y = np.meshgrid(np.arange(size // stride), np.arange(size // stride))
        grid = np.stack((grid_x, grid_y), axis=2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((len(grid), 1), stride))
    grid = np.concatenate(grids)
    stride_values = np.concatenate(strides)
    if output.shape != (1, len(grid), 19):
        raise ValueError(f'Unexpected YOLOX output: {output.shape}')
    decoded = output[0].copy()
    if not np.isfinite(decoded).all() or decoded[:, 4:].min() < 0 or decoded[:, 4:].max() > 1:
        raise ValueError('Expected finite raw YOLOX coordinates and sigmoid probabilities')
    decoded[:, :2] = (decoded[:, :2] + grid) * stride_values
    decoded[:, 2:4] = np.exp(np.clip(decoded[:, 2:4], -20, 20)) * stride_values
    return decoded


class YoloxLight:
    def __init__(self, weights, mode):
        if mode != 'state':
            raise ValueError('YOLOX traffic-light model has no vehicle classes')
        self.names = NAMES
        self.network = cv2.dnn.readNetFromONNX(weights)
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv2.setNumThreads(4)

    def predict(self, image, imgsz=640, conf=.001, iou=.7, max_det=300, classes=None, **kwargs):
        if imgsz != 640:
            raise ValueError('This benchmark adapter is validated only at input 640')
        height, width = image.shape[:2]
        ratio = min(imgsz / width, imgsz / height)
        resized = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_CUBIC)
        padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1.0, swapRB=False, crop=False)
        self.network.setInput(blob)
        decoded = decode_yolox(self.network.forward())
        scores = decoded[:, 4:5] * decoded[:, 5:]
        anchors, native_ids = np.where(scores > conf)
        if classes is not None:
            keep = np.isin(native_ids, classes)
            anchors, native_ids = anchors[keep], native_ids[keep]
        boxes = decoded[anchors, :4].copy()
        boxes[:, :2] -= boxes[:, 2:4] / 2
        confidence = scores[anchors, native_ids]
        selected = cv2.dnn.NMSBoxes(boxes.tolist(), confidence.tolist(), conf, iou)
        selected = np.asarray(selected, dtype=int).reshape(-1)[:max_det]
        boxes = boxes[selected]
        boxes[:, 2:4] += boxes[:, :2]
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        predictions = np.column_stack((boxes, confidence[selected], native_ids[selected])).astype(np.float32)
        tensor = torch.from_numpy(predictions)
        return [SimpleNamespace(boxes=SimpleNamespace(data=tensor, cls=tensor[:, 5]))]
