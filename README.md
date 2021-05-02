# Introduction

This project created to ease features extraction using pretrained models. 

## Installation

1. Install directly from github:
```bash
python -m pip install git+https://github.com/yashenkoxciv/torchvision_features_extraction.git
```

2. Install from cloned repository:
```bash
git clone https://github.com/yashenkoxciv/torchvision_features_extraction.git
cd torchvision_features_extraction
python -m pip install .
```

## Usage

1. Create instance of torchvision_features_extraction.features_extractor.FeaturesExtractor using
torchvision_features_extraction.imagenet.get_imagenet_features_extractor:
```python
from PIL import Image
from torchvision_features_extraction import get_imagenet_features_extractor

fe = get_imagenet_features_extractor('resnet', 'cpu')

image_1 = Image.open('/path/to/image_1.jpg')
image_2 = Image.open('/path/to/image_2.jpg')

# returns features tensor with shape (2, 2048)
features_batch = fe.extract_features([image_1, image_2]) 
```

2. Customize FeaturesExtractor, creating derivatives, for your goals.

## TODO

1. Add models from ReID projects
2. Add models from key points detection