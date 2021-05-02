from setuptools import setup


setup(
    name='torchvision_features_extraction',
    version='0.1',
    description='Package that ease features extraction using torchvision models',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yashenkoxciv/torchvision_features_extraction',
    author='Artem Yaschenko',
    author_email='yashenkoxciv@gmail.com',
    license='MIT',
    packages=['torchvision_features_extraction'],
    install_requires=open('requirements.txt', 'r').read().split('\n')
)
