# Filename: helper.py
# Description: A file that contains
# helper functions used throughout the
# pipeline
# 
# 2021-08-16

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from covid_dataset import ODCovidDataset 
from ensemble import od_ensemble
import torchvision
import torch
import utils

# creating function that gets params
def get_dataloader(data_path):

    # defining params
    params = (
            data_path, 
            "/data/kaggle_data/labels/train_image_level.csv",
            "/data/kaggle_data/raw_dicom/train"
    );

    # getting dataset
    dataset = ODCovidDataset(params, None);

    # creating data loader
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 8,
            shuffle = True,
            num_workers = 16,
            collate_fn = utils.collate_fn);

    # returning data loader
    return data_loader;

def create_model(model_name = None, box_thresh = 0.65, rpn_thresh = 0.7, model_names = [], create_model = None):
    
    # initializing model
    model = None;

    if(model_name == "ensemble"):
        # getting ensemble
        model = od_ensemble(model_names, create_model);

    else:
        # creating model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    pretrained = False,
                    box_score_thresh = box_thresh,
                    rpn_nms_thresh = rpn_thresh
                );

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features;
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2);
            
        # loading model weights
        if(model_name != None):
            model.load_state_dict(torch.load("./{}".format(model_name)));

    # returning model
    return model;

def nightly_create_model(model_name = None, box_thresh = 0.65, rpn_thresh = 0.7):

    # getting backbone
    backbone = torchvision.models.mobilenet_v2(pretrained = False).features;
    
    # specifying number of out channels for backbone
    backbone.out_channels = 1280;

    # creating Anchor Generator
    anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256),),
            aspect_ratios = ((0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)));

    # creating region of interest pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names = [0], output_size = 7,
            sampling_ratio = 2);

    # creating model
    model = FasterRCNN(backbone,
            num_classes = 2, rpn_anchor_generator = anchor_generator, box_roi_pool = roi_pooler);

    # loading model weights
    if(model_name != None):
        model.load_state_dict(torch.load("./{}".format(model_name)));

    # returning model
    return model;

# nightly_create_model();
