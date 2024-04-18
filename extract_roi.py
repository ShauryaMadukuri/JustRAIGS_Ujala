import numpy as np
import os
import pandas as pd
import cv2

from PIL import Image

def preprocess(image_path):
    # Read the image	 
    current_directory = os.getcwd()
    image_path = os.path.join(current_directory, image_path)

    image = cv2.imread(image_path)	   
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image to 608x800	    # Resize the image to 608x800
    resized_image = cv2.resize(image_rgb, (800, 608))	  
    return resized_image

# # Define a function to convert bounding boxes from (xcenter, ycenter, width, height) to (x1, y1, x2, y2)
# def convert_to_x1y1x2y2(boxes,image):
#   h, w = image.size
#   result=[]
#   for box in boxes:
#     xcenter,ycenter,width,height=box
#     # Convert ratios to absolute coordinates
#     x1 = int((xcenter - width / 2) * w)
#     y1 = int((ycenter - height / 2) * h)
#     x2 = int((xcenter + width / 2) * w)
#     y2 = int((ycenter + height / 2) * h)
#     new_box=[x1,y1,x2,y2]
#     result.append(new_box)

#   return result


def merge_boxes(boxes, confidences):
    merged_boxes = []
    merged_confidences = []
    merged_indices = []

    while len(boxes) > 0:
        current_box = boxes[0]
        current_confidence = confidences[0]
        current_index = 0

        for i in range(1, len(boxes)):
            intersection_x1 = np.maximum(current_box[0], boxes[i][0])
            intersection_y1 = np.maximum(current_box[1], boxes[i][1])
            intersection_x2 = np.minimum(current_box[2], boxes[i][2])
            intersection_y2 = np.minimum(current_box[3], boxes[i][3])

            intersection_area = np.maximum(0, intersection_x2 - intersection_x1 + 1) * np.maximum(0, intersection_y2 - intersection_y1 + 1)
            area_current_box = (current_box[2] - current_box[0] + 1) * (current_box[3] - current_box[1] + 1)
            area_new_box = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1)

            iou = intersection_area / (area_current_box + area_new_box - intersection_area)

            if iou > 0:
                # Merge boxes
                current_box = [np.min([current_box[0], boxes[i][0]]), np.min([current_box[1], boxes[i][1]]),
                               np.max([current_box[2], boxes[i][2]]), np.max([current_box[3], boxes[i][3]])]
                current_confidence = max(current_confidence, confidences[i]) + min(current_confidence, confidences[i]) / 2
                current_index = i

        merged_boxes.append(current_box)
        merged_confidences.append(current_confidence)
        merged_indices.append(current_index)

        boxes = np.delete(boxes, current_index, axis=0)
        confidences = np.delete(confidences, current_index)

    return merged_boxes, merged_confidences, merged_indices

def help_extract(resized_image, x1, y1, x2, y2):
    # Calculate resizing factors
    height_ratio = resized_image.shape[0] / 608
    width_ratio = resized_image.shape[1] / 800

    # Scale the bounding box coordinates to match the original image size
    x1_original = int(x1 * width_ratio)
    y1_original = int(y1 * height_ratio)
    x2_original = int(x2 * width_ratio)
    y2_original = int(y2 * height_ratio)

    # Ensure that the coordinates are within the image bounds
    x1_original = max(0, min(x1_original, resized_image.shape[1]))
    y1_original = max(0, min(y1_original, resized_image.shape[0]))
    x2_original = max(0, min(x2_original, resized_image.shape[1]))
    y2_original = max(0, min(y2_original, resized_image.shape[0]))

    # Extract the region of interest (ROI) from the resized image
    roi = resized_image[y1_original:y2_original, x1_original:x2_original]

    return roi


# def crop_bbox(image, x1, y1, x2, y2):
#     """
#     Crop bounding box from the image based on parameters.
    
#     Args:
#     - image: The resized image from which to crop the bounding box.
#     - x1, y1, x2, y2: Coordinates of the bounding box.
    
#     Returns:
#     - Cropped bounding box image.
#     """
#     # Convert coordinates to integers
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     cropped_bbox = image.crop((x1, y1, x2, y2))
#     return cropped_bbox



def extract_roi_single(model, image_path):
    original_filename = os.path.basename(image_path)
    filename_without_extension = os.path.splitext(original_filename)[0]

    resized_image = preprocess(image_path)
    results = model.predict(resized_image, conf=0.26)
    merged_boxes, merged_confidences, merged_indices = merge_boxes(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy())
    if len(merged_boxes) == 0:
        print(f"{image_path}_has some error in finding OD ")
        return False, None
    max_conf_index = np.argmax(merged_confidences)
    x1, y1, x2, y2 = merged_boxes[max_conf_index]
    box_width = x2 - x1
    box_height = y2 - y1
    roi = help_extract(resized_image, x1, y1, x2, y2)
    return True, roi

