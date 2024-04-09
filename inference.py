import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import cv2
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
from load_models import get_yolo_model,get_image_model_processor,get_roi_model_pocessor, get_roi_model_processor_task2, get_image_model_processor_task2
from predict_task1 import predict_pipeline_glacoma
from predict_task2 import get_result_task2



# Set the environment variable TRANSFORMERS_OFFLINE to 1
os.environ['TRANSFORMERS_OFFLINE'] = '1'
print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")



def run():
    _show_torch_cuda_info()
    # Get the current working directory
    current_directory = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the models
    yolo_model = get_yolo_model()

    # Load the models for task 1
    roi_processor_t1 , roi_model_t1 = get_roi_model_pocessor()
    image_processor_t1, image_model_t1 = get_image_model_processor()

    # Load the models for task 2
    roi_processor_t2 , roi_model_t2 = get_roi_model_processor_task2()
    image_processor_t2, image_model_t2 = get_image_model_processor_task2()
    

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")

        
        image_path=jpg_image_file_name

        prob_score,pred_int,detected,roi=predict_pipeline_glacoma(image_path,yolo_model,roi_processor_t1,
                                                                  roi_model_t1,image_processor_t1
                                                                  ,image_model_t1,device)
        predicted_probability_RG,predicted_label = prob_score,pred_int


        is_referable_glaucoma_likelihood = predicted_probability_RG
        is_referable_glaucoma = bool(predicted_label)
        if is_referable_glaucoma:
            # Define the local directory path where your model is saved
            
            t2_results=get_result_task2(image_path,roi,roi_processor_t2,
                                        roi_model_t2,image_processor_t2,
                                        image_model_t2,detected)
            features = {
                k: v
                for k, v in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), t2_results)
            }
        else:
            # make all null
            features = None
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
