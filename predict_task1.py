import torch
from PIL import Image
import torch.nn.functional as F
import os
from extract_roi import extract_roi_single
import cv2


# perdict if OD has been detected
def predict_roi(roi,processor,model,device):
  inputs = processor(images=roi, return_tensors="pt").to(device)
  outputs = model(**inputs).logits.squeeze(dim=1)
  prob_score = torch.sigmoid(outputs).item()
  pred = torch.round(torch.sigmoid(outputs))
  pred_label = int(pred.detach().cpu().numpy())

  return prob_score,pred_label

# Predict if image
def predict_image(image_path,processor,model,device):
    current_directory = os.getcwd()
    image_path = os.path.join(current_directory, image_path)
    image=cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image with the provided processor
    inputs = processor(images=image, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1)
    predicted_probability_RG = probs[0][1].item()

    return predicted_probability_RG,predicted_label



def predict_pipeline_glacoma(raw_image_path,yolo_model,roi_processor,roi_model,image_processor,image_model,device='cuda'):

  # detected , roi=extract_roi_single(yolo_model,raw_image_path)
  detected=False
  roi=None
  if detected:
    prob_score,pred_int=predict_roi(roi,roi_processor,roi_model,device)
  else:
    prob_score,pred_int=predict_image(raw_image_path,image_processor,image_model,device)

  return prob_score,pred_int,detected,roi