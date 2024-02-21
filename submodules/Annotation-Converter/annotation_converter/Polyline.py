import numpy as np

class Polyline:
    def __init__(self, label, points=None):
        self.label = label
        if points:
            self.set_polyline_points_as_array(points)
        else:
            self.points = {"x": [], "y": []}
    def get_label(self):
        return self.label

    def get_polyline_points_as_array(self):
        return np.transpose(np.array([self.points["x"], self.points["y"]]))

    def get_polyline_points(self):
        return self.points

    def set_polyline_points_as_array(self, points):
        for point in points:
            self.points["x"].append(point[0])
            self.points["y"].append(point[1])

    def add_point(self, x, y):
        self.points["x"].append(x)
        self.points["y"].append(y)