# Filename: helper.py
# Description: This file implements some helper
# functions to be used by classification files
# 
# 2021.09.22

# importing tools
from covid_models import DenseNet, XceptionNet, ResNet, EfficientNet, InceptionNet, hyperModel, InceptionResNet

# building model dictionary
model_dict = { 
        "dense": DenseNet, 
        "xception": XceptionNet,
        "resnet": ResNet,
        "efficient": EfficientNet,
        "hyper": hyperModel,
        "inception": InceptionNet,
        "inceptionr": InceptionResNet
};

# building weight dictionary
weight_dict = {
        "dense": "DenseNet",
        "xception": "Xception",
        "resnet": "ResNet50",
        "efficient": "EfficientNet",
        "hyper": None,
        "inception": "Inception",
        "inceptionr": "InceptionResNet"
};

# building params dictionary
params_dict = {
        "dense": 0.001,
        "xception": 0.00015,
        "resnet": 0.000055,
        "efficient": 0.00001,
        "hyper": None,
        "inception": 0.000065,
        "inceptionr": 0.00007
};

