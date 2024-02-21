class Ellipse:
    def __init__(self, label, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.label = label

    def get_xywh(self):
        return self.x, self.y, self.width, self.height

    def get_lefttop_rightbottom(self):
        return self.x, self.y, self.x + self.width, self.y + self.height

    def set_lefttop_rightbottom(self, x1, y1, x2, y2):
        self.width = x2 - x1
        self.height = y2 - y1
        self.x = x1
        self.y = y1
