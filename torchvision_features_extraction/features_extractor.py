import torch
import torch.nn as nn
from torchvision import transforms


class FeaturesExtractor:
    def __init__(self, model: nn.Module, transformations: transforms.Compose, device: str):
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        self.transformations = transformations
        self.device = device

    def extract_features(self, input_images: list):
        input_batch = self._prepare_input_image(input_images).to(self.device)
        features = self.model(input_batch)
        return features

    def extract_features_batch(self, input_batch: torch.Tensor):
        features = self.model(input_batch)
        return features

    def _prepare_input_image(self, input_images):
        input_batch_list = [self.transformations(image).unsqueeze(dim=0) for image in input_images]
        input_batch = torch.cat(input_batch_list, dim=0)
        return input_batch



