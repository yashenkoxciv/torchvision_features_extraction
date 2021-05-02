import unittest
import torch.nn as nn
from torchvision_features_extraction import FeaturesExtractor
from torchvision_features_extraction.imagenet import get_imagenet_transformations
from torchvision_features_extraction import get_imagenet_features_extractor

from PIL import Image


class TestFeatureExtractors(unittest.TestCase):
    def test_transformations(self):
        transforms = get_imagenet_transformations(224)
        fe = FeaturesExtractor(nn.Sequential(), transforms, 'cpu')

        image_1 = Image.open('/home/ahab/dataset/101_ObjectCategories/pizza/image_0001.jpg')
        image_2 = Image.open('/home/ahab/dataset/101_ObjectCategories/pizza/image_0002.jpg')

        images_batch = fe._prepare_input_image([image_1, image_2])

        result = all([x == y for x, y in zip(images_batch.shape, (2, 3, 224, 224))])
        self.assertEqual(result, True)

    def test_fe_1(self):
        fe = get_imagenet_features_extractor('resnet', 'cpu')

        image_1 = Image.open('/home/ahab/dataset/101_ObjectCategories/pizza/image_0001.jpg')
        image_2 = Image.open('/home/ahab/dataset/101_ObjectCategories/pizza/image_0002.jpg')

        features_batch = fe.extract_features([image_1, image_2])

        result = all([x == y for x, y in zip(features_batch.shape, (2, 2048))])
        self.assertEqual(result, True)


if __name__ == '__main__':
    unittest.main()
