#!/usr/bin/env python3
"""
Module to initialize Yolo
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    class that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor:
        - model_path: path to where a Darknet Keras model is stored
        - classes_path: path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        - class_t: float representing the box score threshold for the
        initial filtering step
        - nms_t: float representing the IOU threshold for non-max suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            -> outputs: number of outputs (predictions) made by
            the Darknet model
            -> anchor_boxes: number of anchor boxes used for each prediction
            -> 2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
        - model: the Darknet Keras model
        - class_names: a list of the class names for the model
        - class_t: the box score threshold for the initial filtering step
        - nms_t: the IOU threshold for non-max suppression
        - anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [class_name.strip() for class_name in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        - outputs: list of numpy.ndarrays containing the
        predictions from the Darknet model for a single image:
        Each output have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes):
            - grid_height & grid_width => the height and width
            of the grid used for the output
            - anchor_boxes => the number of anchor boxes used
            4 => (t_x, t_y, t_w, t_h)
            1 => box_confidence
            classes => class probabilities for all classes
        - image_size: numpy.ndarray containing the image’s original
        size [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        - boxes: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4) containing the
        processed boundary boxes for each output, respectively:
            - 4 => (x1, y1, x2, y2)
            - (x1, y1, x2, y2) should represent the boundary box
            relative to original image
            - box_confidences: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing
            the box confidences for each output, respectively
            - box_class_probs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing
            the box’s class probabilities for each output, respectively
        """
        boxes = []

        for index, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, a = output.shape
            image_height, image_width = image_size

            box_xy = self.sigmoid(output[:, :, :, :2])

            box_wh = np.exp(output[:, :, :, 2:4])

            anchor = self.anchors.reshape(1, 1,
                                          self.anchors.shape[0],
                                          anchor_boxes, 2)
            box_wh *= anchor[:, :, index, :, :]
            col = np.tile(np.arange(0, grid_width),
                          grid_height).reshape(grid_height, grid_width)
            row = np.tile(np.arange(0, grid_height),
                          grid_width).reshape(grid_width, grid_height).T
            col = col.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=2)
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=2)
            grid = np.concatenate((col, row), axis=3)

            box_xy += grid
            box_xy /= (grid_width, grid_height)
            input_h = self.model.input.shape[2]
            input_w = self.model.input.shape[1]
            box_wh /= (input_w, input_h)
            box_xy -= (box_wh / 2)
            box_xy1 = box_xy
            box_xy2 = box_xy1 + box_wh
            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            box[..., 0] *= image_size[1]
            box[..., 2] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 3] *= image_size[0]

            boxes.append(box)

        confidence = [self.sigmoid(out[..., 4:5]) for out in outputs]
        prob = [self.sigmoid(out[..., 5:]) for out in outputs]

        return ((boxes, confidence, prob))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        - boxes:list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4) containing the
        processed boundary boxes for each output, respectively
        - box_confidences: list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1) containing the
        processed box confidences for each output, respectively
        - box_class_probs: list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, classes) containing
        the processed box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            - filtered_boxes: numpy.ndarray of shape (?, 4) containing all
            of the filtered bounding boxes:
            - box_classes: numpy.ndarray of shape (?,) containing the
            class number that each box in filtered_boxes predicts.
            - box_scores: numpy.ndarray of shape (?) containing the box
            scores for each box in filtered_boxes, respectively
        """
        box = []

        for conf, probs in zip(box_confidences, box_class_probs):
            box.append(conf * probs)

        max_score = [np.max(elem, axis=-1) for elem in box]
        flat = [elem.reshape(-1) for elem in max_score]
        box_final = np.concatenate(flat, axis=-1)
        pos = np.where(box_final >= self.class_t)
        scores = box_final[pos]

        filter = [elem.reshape(-1, 4) for elem in boxes]
        filter = np.concatenate(filter, axis=0)
        filter = filter[pos]

        classes_max = [elem.argmax(axis=-1) for elem in box_class_probs]
        classes_flat = [elem.reshape(-1) for elem in classes_max]
        classes_final = np.concatenate(classes_flat, axis=-1)
        classes = classes_final[pos]

        return (filter, classes, scores)
