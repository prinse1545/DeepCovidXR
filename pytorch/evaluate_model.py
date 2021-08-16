# Filename: evaluate_model.py
# Description: This file evaluates 
# pytorch faster rcnn models
# 
# 2021-08-12

# importing tools
import matplotlib.pyplot as plt;
from engine import evaluate
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import box_iou
from covid_dataset import ODCovidDataset 
from PIL import Image
import utils
import shutil
import numpy
import cv2
import re
import os


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

def create_model(model_name):

    # creating model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained = False,
                box_score_thresh = 0.65,
                rpn_nms_thresh = 0.7
            );

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features;
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2);
        
    # loading model weights
    model.load_state_dict(torch.load("./{}".format(model_name)));

    # returning model
    return model;


def evaluate_model(prefix):


    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device));

    # assuming that model is in root directory
    files = os.listdir(".");
    
    # finding saved models based on prefix
    model_names = [model_name for model_name in files if re.search(r"^{}.*".format(prefix) , model_name)];

    # sorting by epoch
    model_names.sort( key = lambda name: int(name.split("_")[-1]));

    # getting data loader 
    test_loader = get_dataloader("test");

    # avg precision and recall
    AP = [];
    AR = [];

    # loading models and evaluating
    for model_name in model_names:
         
        # creating model
        model = create_model("./{}".format(model_name));

        # fitting model to device
        model = model.to(device);

        # eval 
        model.eval();
        
        # letting user know
        print("evaluating: {}".format(model_name));
 
        # initializing false positives and negatives
        false_p = 0;
        false_n = 0;
        true_p = 0;

        # iterating
        for batch in test_loader:
            # getting predictions
            preds = model(list(image.to(device) for image in batch[0]));

            # initializing list of integers indicating if box has been detected
            for pred, label in zip(preds, batch[1]):
                # converting to numpy
                grnd_trth  = [label["boxes"][indx].numpy() for indx, labl in enumerate(label["labels"]) if labl.numpy() == 1];
                
                # detected array
                detected = numpy.zeros(len(grnd_trth));
                
                # getting boxes predicted
                boxes = pred["boxes"].cpu().detach().numpy();

                # iterating over predictions
                for box in boxes:
                    # boolean that keeps track of detection
                    detects = False;

                    # finding overlap
                    for indx, gt_box in enumerate(grnd_trth):
                        # finding iou
                        iou = box_iou(
                                torch.as_tensor([box], dtype = torch.float), 
                                torch.as_tensor([gt_box], dtype = torch.float)
                        ).numpy()[0][0];

                        # if iou is greater than 0.5 and the box hasn't been detected, mark as detected
                        if(iou >= 0.5 and detected[indx] == 0):
                            detected[indx] = 1;
                            # marking detects as tru
                            detects = True;

                    # if the box doesn't detect anything new then bye
                    if(not detects):
                        false_p = false_p + 1;
                
                # calculating number of detected
                n_detected = numpy.count_nonzero(detected);

                # calculating false negatives and true positives
                false_n = false_n + (len(detected) - n_detected);
                true_p = true_p + n_detected;


        # getting average precision and average recall
        avg_precision = true_p / (true_p + false_p);
        avg_recall = true_p / (true_p + false_n);

        # printing results
        print("AP:", avg_precision, "\nAR:", avg_recall);

        # testing model
        # metrics = evaluate(model, test_loader, device = device);
        # appending metrics
        # AP.append(metrics.coco_eval["bbox"].stats[1]);
        # AR.append(metrics.coco_eval["bbox"].stats[8]);
    
    """
    # creating x
    x = range(len(model_names));

    # plotting
    fig = plt.figure();
    ax = fig.add_subplot(111);

    ax.scatter(x, AP, c = "b", label = "avg precision");
    ax.scatter(x, AR, c = "r", label = "avg recall");

    plt.legend();
    
    # saving plot
    fig.savefig("./graph.png");
    """;

def evaluate_bbs(model_name):

    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device));

    # getting rid of write dir if it does exist
    if(os.path.isdir("./eval_bbs")):
        shutil.rmtree("./eval_bbs");

    # making dir
    os.makedirs("./eval_bbs");

    # getting data loader
    test_loader = get_dataloader("test");

    # creating model
    model = create_model(model_name);

    # fitting model to device
    model = model.to(device);

    # eval 
    model.eval();

    # geting batch
    batch = next(iter(test_loader));
    
    # creating images
    images = list(image.to(device) for image in batch[0]);

    # getting predictions
    preds = model(images);
    
    # getting radians
    theta = numpy.pi * 3 / 2;

    # creating rotation matrix
    r_mat = numpy.array([
        [numpy.cos(theta), -1 * numpy.sin(theta)], 
        [numpy.sin(theta), numpy.cos(theta)]
    ]);

    # drawing boxes
    for indx, (pred, img, targ) in enumerate(zip(preds, images, batch[1])):

        # converting boxes and images
        np_img = numpy.transpose(img.cpu().detach().numpy()).copy();
        boxes = pred["boxes"].cpu().detach().numpy();

        # iterating over predicted boxes
        for box in boxes:

            # getting mins and maxs
            mins = numpy.array(box[0:2]);
            maxs = numpy.array(box[2:]);

            # rotating
            r_mins = numpy.matmul(r_mat, mins);
            r_maxs = numpy.matmul(r_mat, maxs);

            # drawing rectangle
            cv2.rectangle(np_img,
                    (int(r_mins[0]), int(-1 * r_mins[1])),
                    (int(r_maxs[0]), int(-1 * r_maxs[1])),
                    color = (220, 0, 0),
                    thickness = 2
            );

        # iterating over ground truth boxes
        for box, label in zip(targ["boxes"], targ["labels"]):
            # checking labels
            if(label.numpy() == 1):
                # getting numpy box
                box = box.numpy();

                # creating min and max
                mins = numpy.array(box[0:2]);
                maxs = numpy.array(box[2:]);

                # rotating
                r_mins = numpy.matmul(r_mat, mins);
                r_maxs = numpy.matmul(r_mat, maxs);

                # plotting box
                cv2.rectangle(np_img,
                        (int(r_mins[0]), int(-1 * r_mins[1])),
                        (int(r_maxs[0]), int(-1 * r_maxs[1])),
                        color = (0, 220, 0),
                        thickness = 2
                );

        Image.fromarray(np_img.astype(numpy.uint8)).convert("RGB").save("./eval_bbs/{}.png".format(indx));

# evaluate_model("saved_model");
evaluate_bbs("saved_model_50");
