---
license: apache-2.0
tags:
- image-classification
- vision
- fundus
- glaucoma
- REFUGE
widget:
- src: >-
    https://huggingface.co/pamixsun/swinv2_tiny_for_glaucoma_classification/resolve/main/example.jpg
  example_title: fundus image
---
# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

This model utilizes a Swin Transformer architecture and has undergone supervised fine-tuning on retinal fundus images from the [REFUGE challenge dataset](https://refuge.grand-challenge.org/). 
It is specialized in automated analysis of retinal fundus photographs for glaucoma detection. 
By extracting hierarchical visual features from input fundus images in a cross-scale manner, the model is able to effectively categorize each image as either glaucoma or non-glaucoma. Extensive experiments demonstrate that this model architecture achieves state-of-the-art performance on the REFUGE benchmark for fundus image-based glaucoma classification. 
To obtain optimal predictions, it is recommended to provide this model with standardized retinal fundus photographs captured using typical fundus imaging protocols.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** [Xu Sun](https://pamixsun.github.io)
- **Shared by:** [Xu Sun](https://pamixsun.github.io)
- **Model type:** Image classification
- **License:** Apache-2.0

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

The pretrained model provides glaucoma classification functionality solely based on analysis of retinal fundus images. 
You may directly utilize the raw model without modification to categorize fundus images as either glaucoma or non-glaucoma. 
This model is specialized in extracting discriminative features from fundus images to identify glaucoma manifestations. 
However, to achieve optimal performance, it is highly recommended to fine-tune the model on a representative fundus image dataset prior to deployment in real-world applications.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model is specialized in analyzing retinal fundus images, and is trained exclusively on fundus image datasets to classify images as glaucoma or non-glaucoma. 
Therefore, to obtain accurate predictions, it is crucial to only input fundus images when using this model. 
Feeding other types of images may lead to meaningless results. 
In summary, this model expects fundus images as input for glaucoma classification. 
For the best performance, please adhere strictly to this input specification.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import cv2
import torch

from transformers import AutoImageProcessor, Swinv2ForImageClassification

image = cv2.imread('./example.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

processor = AutoImageProcessor.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")
model = Swinv2ForImageClassification.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts either glaucoma or non-glaucoma.
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

```

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]


## Model Card Contact

- pamixsun@gmail.com