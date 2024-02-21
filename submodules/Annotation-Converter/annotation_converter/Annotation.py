class Annotation:
    def __init__(self, image_location, img_width, img_height, bb_list=[], polygon_list=[], ellipse_list=[], polyline_list=[]):
        self.image_name = image_location
        self.img_width = img_width
        self.img_height = img_height
        self.bb_list = bb_list
        self.polygon_list = polygon_list
        self.ellipse_list = ellipse_list
        self.polyline_list = polyline_list

    def get_image_name(self):
        return self.image_name

    def get_img_width(self):
        return self.img_width

    def get_img_height(self):
        return self.img_height

    def get_bounding_boxes(self):
        return self.bb_list

    def add_ellipse(self, ellipse):
        self.ellipse_list.append(ellipse)

    def get_ellipses(self):
        return self.ellipse_list

    def add_polyline(self, polyline):
        self.polyline_list.append(polyline)

    def get_polylines(self):
        return self.polyline_list

    def add_bounding_box(self, bb):
        self.bb_list.append(bb)
    
    def remove_bounding_box(self, bb):
        x, y, w, h = bb.get_x(), bb.get_y(), bb.get_width(), bb.get_height()
        for i, bb_i in enumerate(self.bb_list):
            x_i, y_i, w_i, h_i = bb_i.get_x(), bb_i.get_y(), bb_i.get_width(), bb_i.get_height()
            if x == x_i and y == y_i and w == w_i and h == h_i:
                self.remove_bounding_box_at_index(i)
                break

    def remove_bounding_box_at_index(self, i):
        del self.bb_list[i]
        
    def add_polygon(self, pol):
        self.polygon_list.append(pol)

    def get_polygons(self):
        return self.polygon_list
