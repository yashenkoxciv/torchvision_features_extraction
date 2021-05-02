import torch.nn as nn
from torchvision import models
from torchvision import transforms

from .features_extractor import FeaturesExtractor


def get_imagenet_transformations(input_size: int):
    imagenet_transformations = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return imagenet_transformations


def imagenet_features_extractor(model_name):
    # Thanks to: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50 """
        model_ft = models.resnet50(pretrained=True)
        model_ft.fc = nn.Sequential()
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet """
        model_ft = models.alexnet(pretrained=True)
        model_ft.classifier[6] = nn.Sequential()
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn """
        #model_ft = models.vgg11_bn(pretrained=True)
        model_ft = models.vgg16(pretrained=True)
        model_ft.classifier[6] = nn.Sequential()
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet """
        model_ft = models.squeezenet1_0(pretrained=True)
        model_ft.classifier[1] = nn.Sequential()
        input_size = 224

    elif model_name == "densenet":
        """ Densenet """
        model_ft = models.densenet121(pretrained=True)
        model_ft.classifier = nn.Sequential()
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=True)
        # Handle the auxilary net
        model_ft.AuxLogits.fc = nn.Sequential()
        # Handle the primary net
        model_ft.fc = nn.Sequential()
        input_size = 299

    else:
        raise RuntimeError("Invalid model name " + model_name)

    return model_ft, input_size


def get_imagenet_features_extractor(model_name, device):
    model, input_size = imagenet_features_extractor(model_name)
    transformations = get_imagenet_transformations(input_size)

    fe = FeaturesExtractor(model, transformations, device)

    return fe

