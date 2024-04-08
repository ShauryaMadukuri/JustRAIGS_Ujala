import torch
from PIL import Image


def get_result_task2(image_path,roi_patch,roi_processor_t2,roi_model_t2,image_processor_t2,image_model_t2,detected):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if detected:
    t2_results=predict_t2_roi(roi_patch,roi_processor_t2,roi_model_t2,device)
  else:
    t2_results=predict_t2_image(image_path,image_processor_t2,image_model_t2,device)

  return t2_results



def predict_t2_image(image_path,processor,model,device):
  image = Image.open(image_path)

  # Process the image with the provided processor
  inputs = processor(images=image, return_tensors="pt").to(device)

  outputs = model(**inputs).logits

  prob_predictions = torch.sigmoid(outputs)
  predictions=(prob_predictions>0.4).tolist()[0]
  return predictions

def predict_t2_roi(roi,processor,model,device):
  inputs = processor(images=roi, return_tensors="pt").to(device)
  outputs = model(**inputs).logits

  prob_predictions = torch.sigmoid(outputs)
  predictions=(prob_predictions>0.5).tolist()[0]
  return predictions
