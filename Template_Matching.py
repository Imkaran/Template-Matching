import cv2
import argparse
from glob import glob
from Template import Template
from OpencvColorCode import get_color_codes
from utils import plot_detection, get_filtered_detection, predict

if __name__=='__main__':
    print("Processing....")
    #Parse argument list
    ap = argparse.ArgumentParser()

    ap.add_argument("-match_threshold", "--match_threshold", required=False, default=0.5, type=float,
                    help="Template Matching Threshold")
    ap.add_argument("-nmst", "--NMS_Threshold", required=False, default=0.5, type=float,
                    help="Non-Max-Supression Threshold")
    ap.add_argument("-input_image", "--input_image", required=False, default= "./data/Face.jpg",
                    help="Input Image Path")
    ap.add_argument("-template_directory", "--template_directory", required=False, default= "./templates",
                    help="Input Directory")
    ap.add_argument("-output", "--output_directory", required=False, default="./output",
                    help="Output Directory")

    args = vars(ap.parse_args())
    #####################################################################################

    input_image_path = args["input_image"]
    template_directory = args["template_directory"]
    output_directory = args["output_directory"]
    non_max_suppression_threshold = args["NMS_Threshold"]
    match_threshold = args["match_threshold"]

    colors = get_color_codes()

    templates = []
    label = 1

    for template_image in glob(template_directory+"/*.png"):
        templates.append(Template(image_path=template_image, label=label, color = colors[label-1], match_threshold=match_threshold))
        label += 1
    input_image = cv2.imread(input_image_path)
    ########################################################################

    detection = predict(templates, input_image)
    filtered_detection = get_filtered_detection(detections=detection, non_max_suppression_threshold=non_max_suppression_threshold)
    cv2.imwrite(output_directory+"/image_with_detection.png", plot_detection(input_image.copy(), filtered_detection))
    print ("Output Image saved to ", output_directory)