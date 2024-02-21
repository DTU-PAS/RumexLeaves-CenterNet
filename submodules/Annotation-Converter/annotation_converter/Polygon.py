import numpy as np
import cv2
from annotation_converter.BoundingBox import BoundingBox

class Polygon:
    def __init__(self, label):
        self.points = {"x": [], "y": []}
        self.label = label

    def equals(self, pol):
        if pol.get_label() != self.label:
            return False
        points = pol.get_polygon_points()
        if points["x"] != self.points["x"] or points["y"] != self.points["y"]:
            return False
        return True

    def get_polygon_points_as_array(self):
        return np.transpose(np.array([self.points["x"], self.points["y"]]))

    def get_polygon_points(self):
        return self.points

    def set_polygon_points_as_array(self, points):
        for point in points:
            self.points["x"].append(point[0])
            self.points["y"].append(point[1])

    def set_polygon_points(self, points):
        self.points = points

    def add_point(self, x, y):
        self.points["x"].append(x)
        self.points["y"].append(y)

    def get_label(self):
        return self.label

    def to_bounding_box(self):
        cnt = np.transpose(np.array([self.points["x"], self.points["y"]]))
        x, y, w, h = cv2.boundingRect(cnt)
        return BoundingBox(self.label, x, y, w, h)
