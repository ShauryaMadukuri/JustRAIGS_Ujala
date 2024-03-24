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
print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
import os



# # Set the environment variable HF_HOME to the cache directory
# os.environ['HF_HOME'] = '/cache/huggingface'
# print(f"HF_HOME: {os.environ.get('HF_HOME')}")
# # Set the environment variable TRANSFORMERS_CACHE to the cache directory
# os.environ['TRANSFORMERS_CACHE'] = '/cache/huggingface'


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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
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


def print_contents_of_folder(folder_path):
        print(f"Contents of folder: {folder_path}")
        for item in os.listdir(folder_path):
            print(item)


def run():
    _show_torch_cuda_info()
    # Get the current working directory
    current_directory = os.getcwd()

    # Print the current working directory
    print("Current working directory:", current_directory)

    # Print all folder names and file names in the current directory
    print("Contents of the current directory:")
    for item in os.listdir(current_directory):
        print(item)

    # Print all folder names and file names in the current directory
    print("Contents of the current directory:")
    for item in os.listdir(current_directory):
        print(item)
        if os.path.isdir(os.path.join(current_directory, item)):
            print_contents_of_folder(os.path.join(current_directory, item))

    
    print('Starting the model loading process...')
    # Folder to join
    folder_to_join = "swinv2_tiny_model"

    # Join the folder to the current working directory
    model_directory = os.path.join(current_directory, folder_to_join)

    # Print the model_directory path
    print("model_directory =", model_directory)

    # # List files in the model directory
    # print('this is model loading')

    # files_in_model_directory = os.listdir(model_directory)

    # # Print the files
    # print("Files in model_directory:")
    # for file in files_in_model_directory:
    #     print(file)

    # Load the image processor if needed
    processor = AutoImageProcessor.from_pretrained(model_directory)

    # Load the model checkpoint
    model = Swinv2ForImageClassification.from_pretrained(model_directory)
    print('Model 1 loaded successfully')

    # Path of the checkpoint relative to the current directory
    checkpoint_path_relative = 'checkpoints/swin_finetunetask11.pth'

    # Join the checkpoint path with the current working directory
    checkpoint_path = os.path.join(current_directory, checkpoint_path_relative)

    # Print the checkpoint path
    print("checkpoint_path =", checkpoint_path)



    model_directory_2 = "swinv2_tiny_model"

    model_directory_2 = os.path.join(current_directory, model_directory_2)

    print("model_directory_2 =", model_directory_2)

    # Load the image processor if needed
    processor_2 = AutoImageProcessor.from_pretrained(model_directory_2)

    # Load the model checkpoint
    model_2 = Swinv2ForImageClassification.from_pretrained(model_directory_2)

    checkpoint_path_2 = 'checkpoints/swin_finetunetask219.pth'

    checkpoint_path_2 = os.path.join(current_directory, checkpoint_path_2)

    print("checkpoint_path_2 =", checkpoint_path_2)

    print('Model 2 loaded successfully')
    

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
