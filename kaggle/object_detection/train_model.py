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
from engine import train_one_epoch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from covid_dataset import ODCovidDataset 
from helper import get_dataloader, create_model, nightly_create_model
from evaluate_model import evaluate_model
import numpy


def create_frcnn():

    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device))

    # creating model
    model = create_model();

    # attaching model to device
    model = model.to(device);

    # getting data loaders
    train_loader = get_dataloader("/data/kaggle_data/od_split_formatted/train/coarse");
    valid_loader = get_dataloader("/data/kaggle_data/od_split_formatted/valid/coarse");
    test_loader = get_dataloader("/data/kaggle_data/od_split_formatted/test");

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.0002,
                                momentum=0.9)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=100,
                                                   gamma=0.001)
    # defining number of epochs
    num_epochs = 101;

    # initializing metrics array
    metrics = numpy.zeros(shape = (num_epochs, 3));

    # training
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq = 100);
        # update the learning rate
        lr_scheduler.step();

        # evaluate on the test dataset
        ap, ar, fs, _ = evaluate_model(model, valid_loader, device);

        # saving evaluation metrics
        metrics[epoch][0] = ap;
        metrics[epoch][1] = ar;
        metrics[epoch][2] = fs;

        # saving checkpoint
        if(epoch % 5 == 0 and epoch != 0):
            torch.save(model.state_dict(), "two_resnet50_coarse/saved_model_{}".format(epoch));
            # letting user know of success
            print("saved checkpoint");
    
    # testing model
    evaluate_model(model, test_loader, device);
    
    # saving metrics
    numpy.save("two_resnet50_coarse/metrics.npy", metrics);
    
    print("saved model");
    
create_frcnn();
