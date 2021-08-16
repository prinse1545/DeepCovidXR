# Filename: create_model.py
# Description: This file is used to create
# the backbone model for faster rcnn as well as the
# faster rcnn model itself. Depending on parameters
# passed in.
# 
# 2021-08-06

# Importing tools
import torchvision
import torch
from engine import train_one_epoch, evaluate
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from covid_dataset import ODCovidDataset 
import utils
import numpy


# creating function that gets params
def get_dataloader(_type):

    # defining params
    params = (
            "/data/kaggle_data/od_formatted/{}".format(_type), 
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

def create_frcnn():

    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device))


    # creating model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained = False,
            );

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features;
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2);

    # loading model weights
    # model.load_state_dict(torch.load("./{}".format("saved_model_100")));

# attaching model to device
    model = model.to(device);

    # getting data loaders
    train_loader = get_dataloader("train");
    valid_loader = get_dataloader("valid");
    test_loader = get_dataloader("test");

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.0002,
                                momentum=0.9, weight_decay=0.000)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=100,
                                                   gamma=0.001)

    
    # defining number of epochs
    num_epochs = 100;

    # training
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq = 100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)

        # saving checkpoint
        if(epoch % 10 == 0 and epoch != 0):
            torch.save(model.state_dict(), "./saved_model_{}".format(epoch));
            # letting user know of success
            print("saved checkpoint");
    
    # testing model
    evaluate(model, test_loader, device = device);

    
    print("saved model");
    
create_frcnn();
