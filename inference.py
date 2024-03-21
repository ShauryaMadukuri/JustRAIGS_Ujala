import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
from transformers import AutoModelForImageClassification, AutoConfig

import os

# Set the environment variable TRANSFORMERS_OFFLINE to 1
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def get_result_task1(processor,model,checkpoint_path,image_path):
    model.load_state_dict(torch.load(checkpoint_path))
    # Assuming you have a validation DataLoader named val_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #   image = cv2.imread(image_path)
    #   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.open(image_path)

    # Process the image with the provided processor
    inputs = processor(images=image, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1)
    predicted_probability_RG = probs[0][1].item()

    return predicted_probability_RG,predicted_label


def get_result_task2(processor,model,checkpoint_path,image_path):
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model.load_state_dict(torch.load(checkpoint_path))

    # Assuming you have a validation DataLoader named val_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Image processing
    #   image = cv2.imread(image_path)
    #   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.open(image_path)

    # Process the image with the provided processor
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs).logits

    prob_predictions = torch.sigmoid(outputs)
    predictions=(prob_predictions>0.4).tolist()[0]
    return predictions





def run():
    _show_torch_cuda_info()
    model_directory = "/swinv2_tiny_model"
    # Load the configuration file
    config = AutoConfig.from_pretrained(model_directory)

    # Load the image processor if needed
    processor = AutoImageProcessor.from_pretrained(model_directory)

    # Load the model checkpoint
    model = AutoModelForImageClassification.from_pretrained(model_directory, config=config)

    checkpoint_path = '/checkpoints/swin_finetunetask11.pth'

    model_directory_2 = "/swinv2_tiny_model"
    # Load the configuration file
    config_2 = AutoConfig.from_pretrained(model_directory_2)

    # Load the image processor if needed
    processor_2 = AutoImageProcessor.from_pretrained(model_directory_2)

    # Load the model checkpoint
    model_2 = AutoModelForImageClassification.from_pretrained(model_directory_2, config=config_2)

    checkpoint_path_2 = '/checkpoints/swin_finetunetask219.pth'

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")

        # # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        # image = Image.open(jpg_image_file_name)
        # numpy_array = numpy.array(image)
        
        image_path=jpg_image_file_name

        result_task1 = get_result_task1(processor,model,checkpoint_path,image_path)
        predicted_probability_RG,predicted_label = result_task1


        is_referable_glaucoma_likelihood = predicted_probability_RG
        is_referable_glaucoma = predicted_label
        if is_referable_glaucoma:
            # Define the local directory path where your model is saved
            
            result_task2 = get_result_task2(processor_2,model_2,checkpoint_path_2,image_path)
            features = {
                k: v
                for k, v in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), result_task2)
            }
        else:
            # make all null
            features = {
                k: None
                for k in DEFAULT_GLAUCOMATOUS_FEATURES.keys()
            }
        ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
