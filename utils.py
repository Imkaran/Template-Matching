import cv2
import numpy as np
from tqdm import tqdm

def predict(templates, input_image):

    '''
    Find the detection on input image by matching template
    :param templates: List of Class Objects
    :param input_image: CV2 image array
    :return detection: list of dictionary
    '''

    detection = []
    for template in tqdm(templates, desc="Prediction: "):
        #caculate score value for each pixel
        template_matching = cv2.matchTemplate(input_image, template.template_image, cv2.TM_CCOEFF_NORMED)

        #get the location of pixels who score value >= match_threshold
        match_locations = np.where(template_matching >= template.match_threshold)

        for (x, y) in zip(match_locations[1], match_locations[0]):
            match = {
                "TOP_LEFT_X": x,
                "TOP_LEFT_Y": y,
                "BOTTOM_RIGHT_X": x + template.template_width,
                "BOTTOM_RIGHT_Y": y + template.template_height,
                "COLOR": template.color,
                "LABEL": template.label,
                "MATCH_VALUE": template_matching[y, x]
            }

            detection.append(match)

    return detection


def get_iou(a, b, epsilon=1e-5):
    '''
    Calculate the intersection over union between two bounding boxes using the formula:
    IOU = (Area of intersection / Area of Union)
    :param a: bounding box coordinate of first box [x1, y1, x2, y2]
    :param b: bounding box coordinate of second box [x1, y1, x2, y2]
    :param epsilon: Small value to prevent from divide by zero error
    :return: float(iou) value
    '''
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a["TOP_LEFT_X"], b["TOP_LEFT_X"])
    y1 = max(a["TOP_LEFT_Y"], b["TOP_LEFT_Y"])
    x2 = min(a["BOTTOM_RIGHT_X"], b["BOTTOM_RIGHT_X"])
    y2 = min(a["BOTTOM_RIGHT_Y"], b["BOTTOM_RIGHT_Y"])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a["BOTTOM_RIGHT_X"] - a["TOP_LEFT_X"]) * (a["BOTTOM_RIGHT_Y"] - a["TOP_LEFT_Y"])
    area_b = (b["BOTTOM_RIGHT_X"] - b["TOP_LEFT_X"]) * (b["BOTTOM_RIGHT_Y"] - b["TOP_LEFT_Y"])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def get_filtered_detection(detections, non_max_suppression_threshold=0.4, score_key="MATCH_VALUE"):
    '''
    filter detection by using non_max_supression threshold
    :param detections: list of dictionary
    :param non_max_suppression_threshold: float
    :param score_key: String
    :return filtered_objects: list of dictionary
    '''

    #sort the detection list as per the descending order of score_key
    sorted_object = sorted(detections, key=lambda det : det[score_key], reverse=True)
    filtered_objects = []

    #filter detection by non-max-supression
    for detection in tqdm(sorted_object, desc="object filteration: "):
        overlap = False
        for filtered_object in filtered_objects:
            iou = get_iou(detection, filtered_object)
            if iou > non_max_suppression_threshold:
                overlap = True
                break

        if not overlap:
            filtered_objects.append(detection)

    return filtered_objects


def plot_detection(image_with_detections, detections):
    '''
    Plot detected bounding boxes on image
    :param image_with_detections: cv2 raw image array
    :param detections: list of dictionary contains bounding boxes
    :return image_with_detections: cv2 image array with detections
    '''
    for detection in detections:

        cv2.rectangle(image_with_detections,
                      (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
                      (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
                      detection["COLOR"], 2)


    return image_with_detections

