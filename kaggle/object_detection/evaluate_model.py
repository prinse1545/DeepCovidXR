# Filename: evaluate_model.py
# Description: This file evaluates 
# pytorch faster rcnn models
# 
# 2021-08-12

# importing tools
from matplotlib import cm
import matplotlib.pyplot as plt;
from matplotlib.ticker import LinearLocator
import torch
from torchvision.ops.boxes import box_iou
from PIL import Image
from helper import get_dataloader, create_model, nightly_create_model
import random
import shutil
import numpy
import cv2
import re
import os


def evaluate_model(model, data_loader, device):

    # eval
    model.eval();

    # initializing false positives and negatives
    false_p = 0;
    false_n = 0;
    true_p = 0;

    # initialzing TIDE metrics
    tide = {
        "loc": 0,
        "dup": 0,
        "bkgd": 0,
        "miss": 0
    };

    # iterating
    for batch in data_loader:
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
                # boolean that tracks tide metrics
                tide_failure_accounted = False;

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
                    elif(iou >= 0.5 and detected[indx] == 1):
                        # marking duplicate error
                        tide["dup"] = tide["dup"] + 1;
                        tide_failure_accounted = True;
                    elif(iou > 0):
                        # marking location error
                        tide["loc"] = tide["loc"] + 1;
                        tide_failure_accounted = True;
                    else:
                        # marking background error
                        tide["bkgd"] = tide["bkgd"] + 1;
                        tide_failure_accounted = True;

                # if the box doesn't detect anything new then bye
                if(not detects):
                    false_p = false_p + 1;
                    if(not tide_failure_accounted):
                        tide["miss"] = tide["miss"] + 1;
                
            # calculating number of detected
            n_detected = numpy.count_nonzero(detected);

            # calculating false negatives and true positives
            false_n = false_n + (len(detected) - n_detected);
            true_p = true_p + n_detected;


    # print(true_p, false_p, false_n)
    # getting average precision and average recall
    avg_precision = true_p / (true_p + false_p) if true_p + false_p > 0 else 0;
    avg_recall = true_p / (true_p + false_n) if true_p + false_n > 0 else 0;
    f_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0;

    # printing results
    print("AP:", avg_precision, "\nAR:", avg_recall, "\nF-Score:", f_score, "\ntide:", tide);
    return avg_precision, avg_recall, f_score, tide;


def get_metrics(model_name):
    
    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device));

    # getting data loader
    test_loader = get_dataloader("/data/kaggle_data/od_test/test");

    # creating model
    model = create_model(model_name,
            model_names = [("resnet50_coarse/saved_model_55", 0.7, 0.45),
                ("resnet50_fine/saved_model_80", 0.75, 0.45)],
            create_model = create_model);

    # fitting model to device
    model = model.to(device);

    # evaluating model
    return evaluate_model(model, test_loader, device);


def create_tide_graph(model_name):

    # getting tide metrics
    _, _, tide = get_metrics(model_name);

    # getting labels, data, and explode params
    labels = tide.keys();
    sizes = [tide[key] for key in tide.keys()];
    
    # creating plot
    fig, ax = plt.subplots();
    bars = ax.bar(labels, sizes);
    
    # creating color list
    colors = ["#F1C40F", "#3498DB", "#2ECC71", "#34495E"];

    # iterating over bars to set colors
    for indx, bar in enumerate(bars):
        # setting color
        bar.set_color(colors[indx]);

    ax.set_title("TIDE Chart");

    # saving fig
    fig.savefig("tide_chart.png");

def create_finetuning_graph(model_name):

    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device));

    # getting data loader
    test_loader = get_dataloader("test");

    # defining collecting lists
    threshs = [];
    rnms = [];
    APs = [];
    ARs = [];

    # iterating over different iou thresholds
    for iou in range(0, 100, 10):

        # getting threshold
        thresh = iou / 100;

        # iterating over different region proposal network rnms
        for rpn in range(0, 100, 10):
 
            # getting rnm
            rnm = rpn / 100;

            # creating model
            model = create_model(model_name, thresh, rnm);

            # fitting model to device
            model = model.to(device);

            # evaluating
            ap, ar, _ = evaluate_model(model, test_loader, device);

            print("iou", thresh, "rnm", rnm);
            # getting APs
            APs.append(ap);
            ARs.append(ar);

            # collecting data
            threshs.append(thresh);
            rnms.append(rnm);

    
    # getting fig and axes 
    fig, ax = plt.subplots(subplot_kw = { "projection": "3d" });

    # plotting
    img = ax.scatter(threshs, rnms, APs, c = ARs, cmap = plt.hot());

    # naming axes
    ax.set_xlabel("IoU Positive Threshold");
    ax.set_ylabel("Overlap Joining Threshold");
    ax.set_zlabel("Average Precision");
    # Add a color bar which maps values to colors.
    fig.colorbar(img, shrink=0.5, aspect=5)

    # saving
    fig.savefig("finetune.png");

def draw_pred_bbs(models):

    # enabling GPU if available
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu");
    print("using {}".format(device));

    # getting rid of write dir if it does exist
    if(os.path.isdir("./eval_bbs")):
        shutil.rmtree("./eval_bbs");

    # making dir
    os.makedirs("./eval_bbs");

    # getting data loader
    test_loader = get_dataloader("/data/kaggle_data/od_split_formatted/test");

    # geting batch
    batch = next(iter(test_loader));
    
    # creating images
    images = list(image.to(device) for image in batch[0]);

    # initializing preds array
    preds = [];

    for model in models:

        # creating model
        model = create_model(model[0], model_names = model[3], create_model = create_model);

        # fitting model to device
        model = model.to(device);

        # eval
        model.eval();

        # getting predictions
        preds.append(model(images));
    
    # getting radians
    theta = numpy.pi * 3 / 2;

    # creating rotation matrix
    r_mat = numpy.array([
        [numpy.cos(theta), -1 * numpy.sin(theta)], 
        [numpy.sin(theta), numpy.cos(theta)]
    ]);

    # drawing boxes
    for indx, (preds_tuple, img, targ) in enumerate(zip(zip(*preds), images, batch[1])):
        
        # converting boxes and images
        np_img = numpy.transpose(img.cpu().detach().numpy()).copy();
    
        # iterating over prediction tuple
        for i, pred in enumerate(preds_tuple):
            boxes = pred["boxes"].cpu().detach().numpy();
            scores = pred["scores"].cpu().detach().numpy();

            # iterating over predicted boxes
            for box, score in zip(boxes, scores):

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

                # annotating box
                cv2.putText(np_img, 
                        "Opacity {}: {}%".format(i, round(score * 100, 2)), 
                        (int(r_mins[0]), int(-1 * r_mins[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 0, 0), 2);


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

                # annotating box
                cv2.putText(np_img, 
                        "Ground Truth", 
                        (int(r_mins[0]), int(-1 * r_mins[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2);

        # saving image
        Image.fromarray(np_img.astype(numpy.uint8)).convert("RGB").save("./eval_bbs/{}.png".format(indx));

def create_epoch_graph(metrics_path):

    # reading in metrics from saved numpy
    metrics = numpy.load(metrics_path);
    # transposing
    metrics = numpy.transpose(metrics);
    # creating x axis
    x = list(range(len(metrics[0])));
    
    # plotting
    plt.scatter(x, metrics[0], color = "red", label = "Avg Precision");
    plt.scatter(x, metrics[1], color = "blue", label = "Avg Recall");
    plt.scatter(x, metrics[2], color = "purple", label = "F1 Score");

    # changing ticks
    plt.xticks(x[::5]);

    # setting legend
    plt.legend(title = "Metrics");

    # title and axes
    plt.xlabel("Epoch");
    plt.title("Metrics Over Epochs");

    # setting grid lines
    plt.gca().xaxis.grid(True);

    # saving 
    plt.savefig("resnet50_fine_epoch_graph.png");

# draw_pred_bbs([("ensemble", 0.7, 0.45, [("resnet50_coarse/saved_model_55", 0.7, 0.45), ("resnet50_fine/saved_model_80", 0.75, 0.45)]);
# create_finetuning_graph("resnet50_bone_2/saved_model_45");
get_metrics("ensemble");
# create_epoch_graph("resnet50_fine/metrics.npy");
