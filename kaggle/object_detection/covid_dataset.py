# Filename: covid_dataset.py
# Description: This file implements
# the covid dataset to be used by pytorch
# faster rcnn
#
# 2021-08-05

# importing tools
import os
import numpy
import pandas
import json
import pydicom
from PIL import Image
import torch


# defining object detection data set class
class ODCovidDataset(torch.utils.data.Dataset):
    def __init__(self, loc_info, transforms):
        self.root = loc_info[0];
        self.transforms = transforms;

        # getting images
        self.imgs = os.listdir(loc_info[0]);
        # getting labels
        self.labels = pandas.read_csv(loc_info[1]);
        # getting dicom path
        self.dicom_dir = loc_info[2];

    def __getitem__(self, index):
        
        # saving img path
        img_path = os.path.join(self.root, self.imgs[index]);

        # getting image and converting to rgb
        img = Image.open(img_path).convert("RGB");
        
        # getting kaggle id 
        kaggle_id = "{}_image".format(os.path.splitext(self.imgs[index].split("-")[-1])[0]);        
        
        # getting bounding boxes
        boxes = self.labels[self.labels["id"].str.contains(kaggle_id)].iloc[0].loc["boxes"];

        # getting boxes in json not string
        if(boxes == boxes):
            boxes = json.loads(boxes.replace("'", "\""));
        else:
            boxes = [];
    
        final_boxes = [];
        
        # getting radians
        theta = numpy.pi / 2;

        # creating rotation matrix
        r_mat = numpy.array([
            [numpy.cos(theta), -1 * numpy.sin(theta)], 
            [numpy.sin(theta), numpy.cos(theta)]
        ]);

        for box in boxes:
            # getting mins
            mins = numpy.array([box["x"], box["y"]]);

            # getting maxs
            maxs = numpy.array([box["x"] + box["width"], box["y"] + box["height"]]);

            # rotating
            r_mins = numpy.matmul(r_mat, mins);
            r_maxs = numpy.matmul(r_mat, maxs);

            # converting coordinates
            final_boxes.append(
                    [
                        r_mins[0] * -1024, 
                        r_mins[1] * 1024, 
                        r_maxs[0] * -1024, 
                        r_maxs[1] * 1024
                    ]
            );

        # print(final_boxes) 

        #final_boxes = final_boxes if len(final_boxes) > 0 else [[100.0, 100.0, 150.0, 150.0]]; 
        # converting boxes to tensors
        final_boxes = torch.as_tensor(final_boxes, dtype = torch.float32);
        # one class for labels so we can put all ones
        box_labels = torch.ones(len(boxes), dtype = torch.int64);
        
        # image id
        image_id = torch.tensor([index]);

        # creating final object
        target = {};

        target["boxes"] = final_boxes if len(final_boxes) > 0 else torch.as_tensor([[100.0, 100.0, 150.0, 150.0]]);
        target["labels"] = box_labels if len(final_boxes) > 0 else torch.zeros(1, dtype = torch.int64);
        target["image_id"] = image_id;
        target["iscrowd"] = torch.zeros(len(target["boxes"]), dtype = torch.int64);

        # getting area
        area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0]);
        # print("area", area);
        #print(target["iscrowd"], len(target["boxes"]));
        target["area"] = area;
        #print(len(final_boxes) == 0, target["boxes"], target["labels"], target["area"]);
        # applying transformations if needed
        if self.transforms is not None:
            img, target = self.transforms(img, target);
        # making img array
        img = numpy.transpose(numpy.asarray(img));

        # returning img with its target at index index
        return torch.as_tensor(img, dtype = torch.float32), target;

    def __len__(self):

        # returning length of data set
        return len(self.imgs);


