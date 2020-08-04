import cv2

class Template:

    def __init__(self, image_path, label, color, match_threshold):

        self.image_path = image_path
        self.label = label
        self.match_threshold = match_threshold
        self.color = color
        self.template_image = cv2.imread(image_path)
        self.template_height, self.template_width = self.template_image.shape[0], self.template_image.shape[1]
