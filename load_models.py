import os
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch
import torch.nn as nn
import cv2
import ultralytics
from ultralytics import YOLO


current_directory = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ROI EXTRACTION 

def get_yolo_model():
    yolo_model_path = "yolov8_model_weights\best.pt"
    yolo_model_path = os.path.join(current_directory, yolo_model_path)
    yolo_model = YOLO(yolo_model_path)
    return yolo_model



# TASK 1 Models / Glaucoma Detection


def get_roi_model_pocessor():
    swin_roi_glauc_model_path = "swinv2_scratch_model"
    swin_roi_glauc_model_path = os.path.join(current_directory, swin_roi_glauc_model_path)

    # Load the image processor if needed
    roi_processor = AutoImageProcessor.from_pretrained(swin_roi_glauc_model_path)

    # Load the model checkpoint
    roi_model = Swinv2ForImageClassification.from_pretrained(swin_roi_glauc_model_path)
    roi_model.classifier = nn.Linear(roi_model.classifier.in_features, 1)


    roi_checkpoint_path = 'checkpoints/task1/roi/roi_swin_phase2_finetunetask115.pth'
    roi_checkpoint_path= os.path.join(current_directory, roi_checkpoint_path)
    
    roi_model.load_state_dict(torch.load(roi_checkpoint_path))

    roi_model.to(device)
    return roi_processor, roi_model


def get_image_model_processor():
    swin_full_image_glauc_model_path = "swinv2_tiny_model"
    swin_full_image_glauc_model_path = os.path.join(current_directory, swin_full_image_glauc_model_path)

    # Load the image processor if needed
    image_task1_processor = AutoImageProcessor.from_pretrained(swin_full_image_glauc_model_path)

    # Load the model checkpoint
    image_task1_model = Swinv2ForImageClassification.from_pretrained(swin_full_image_glauc_model_path)

    image_task1_model_checkpoint_path = 'checkpoints/task1/image/swin_phase2_finetunetask15.pth'
    image_task1_model_checkpoint_path = os.path.join(current_directory, image_task1_model_checkpoint_path)

    image_task1_model.load_state_dict(torch.load(image_task1_model_checkpoint_path))
    image_task1_model.to(device)
    return image_task1_processor, image_task1_model


# TASK 2 Models / multi_label_classification

def get_roi_model_processor_task2():
    swin_roi_multi_label_model_path = "swinv2_scratch_model"
    swin_roi_multi_label_model_path = os.path.join(current_directory, swin_roi_multi_label_model_path)

    # Load the image processor if needed
    roi_processor_task2 = AutoImageProcessor.from_pretrained(swin_roi_multi_label_model_path)

    # Load the model checkpoint
    roi_model_task2 = Swinv2ForImageClassification.from_pretrained(swin_roi_multi_label_model_path)
    roi_model_task2.classifier = nn.Linear(roi_model_task2.classifier.in_features, 10)


    roi_checkpoint_path_task2 = 'checkpoints/task2/roi/roi_swin_finetunetask220.pth'
    roi_checkpoint_path_task2 = os.path.join(current_directory, roi_checkpoint_path_task2)

    roi_model_task2.load_state_dict(torch.load(roi_checkpoint_path_task2))

    roi_model_task2.to(device)
    return roi_processor_task2, roi_model_task2


# Task 2 Image Model

def get_image_model_processor_task2():
    swin_full_image_multi_label_model_path = "swinv2_tiny_model"
    swin_full_image_multi_label_model_path = os.path.join(current_directory, swin_full_image_multi_label_model_path)

    # Load the image processor if needed
    image_task2_processor = AutoImageProcessor.from_pretrained(swin_full_image_multi_label_model_path)

    # Load the model checkpoint
    image_task2_model = Swinv2ForImageClassification.from_pretrained(swin_full_image_multi_label_model_path)
    image_task2_model.classifier = nn.Linear(image_task2_model.classifier.in_features, 10)

    image_task2_model_checkpoint_path = 'checkpoints/task2/image/swin_phase2_finetunetask211.pth'
    image_task2_model_checkpoint_path = os.path.join(current_directory, image_task2_model_checkpoint_path)

    image_task2_model.load_state_dict(torch.load(image_task2_model_checkpoint_path))
    image_task2_model.to(device)
    return image_task2_processor, image_task2_model