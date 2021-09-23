# Filename: playground.py
# Description: This file serves as a playground
# for the object detection part of the kaggle
# competition
# 
# 2021-07-12

# importing tools
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import pydicom
import pandas
import random
import numpy
import cv2
import json
import os
import math

def draw_bounding_bs():
    # This function takes 100 random images and 
    # draws their bounding boxes

    # dicom path
    dicom_path = "/data/kaggle_data/raw_dicom";

    # getting filenames
    filenames = [];

    for _, _, files in os.walk(os.path.join(dicom_path, "train")):
        filenames.extend(files);

    # getting random numbers for filenames
    # file_indices = random.sample(range(0, len(filenames)), 100);
    file_indices = [5225, 435, 1626, 4929, 3643, 1560, 1954, 4394, 3982, 2740, 3255, 6037, 2078, 2331, 844, 2408, 4925, 5654, 2885, 2666, 5786, 783, 126, 2362, 1675, 4228, 3586, 2865, 5621, 1972, 3815, 6256, 3863, 1125, 647, 3298, 1097, 5652, 4602, 3581, 3501, 3187, 344, 1117, 1144, 4585, 4981, 1804, 4740, 4550, 59, 3777, 4821, 6238, 3292, 4723, 1342, 5603, 5154, 6152, 4473, 841, 2317, 2656, 2492, 4018, 434, 4236, 6158, 4949, 1630, 829, 5606, 2071, 4620, 4969, 4893, 4625, 3638, 2804, 4244, 3938, 1430, 1376, 2704, 4995, 838, 5008, 5871, 1558, 5898, 1356, 1321, 1624, 2252, 3001, 850, 321, 4809, 1864]

    # getting filenames to draw
    draw_filenames = [filenames[indx] for indx in file_indices];
    print(draw_filenames);
    draw_filenames.append("21acb15b9ee4.dcm");
    # getting training image bounding box data
    image_bb = pandas.read_csv("/data/kaggle_data/labels/train_image_level.csv");

    if(os.path.isdir("/data/bb_imgs")):
        shutil.rmtree("/data/bb_imgs");
        
    os.makedirs("/data/bb_imgs");
    # walking again
    for root, dirs, files in os.walk(os.path.join(dicom_path, "train")):
        # iterating over files
        for _file in files:

            # checking if one of selected files
            if(_file in draw_filenames):
                # getting file path
                f_path = os.path.join(root, _file);

                # reading in dicom
                xray_dicom = pydicom.filereader.dcmread(f_path);

                # getting width amd height
                width = xray_dicom.pixel_array.shape[1];
                height = xray_dicom.pixel_array.shape[0];

                # applying volume of interest look up table colors to get opacity in
                # accordance with dicom image format
                xray = apply_voi_lut(xray_dicom.pixel_array, xray_dicom);

                # fixing inversion stuff if needed
                if(xray_dicom.PhotometricInterpretation == "MONOCHROME1"):
                    xray = numpy.amax(xray) - xray;

                # normalizing
                xray = xray - numpy.min(xray);
                xray = xray / numpy.max(xray);

                # converting to 8 bit unsigned integer (from gray scale 0 to 1)
                xray = (xray * 255).astype(numpy.uint8);

                # resizing
                xray = resize(xray, (224, 224), anti_aliasing = True);

                # getting kaggle id 
                kaggle_id = "{}_image".format(os.path.splitext(_file)[0]);

                # getting bounding boxes
                boxes = image_bb[image_bb["id"].str.contains(kaggle_id)].iloc[0].loc["boxes"];

                # filtering out Nans
                if(boxes == boxes):
                    # getting json version of string
                    j_boxes = json.loads(boxes.replace("'", "\""));

                    # iterating over boxes
                    for box in j_boxes:
                        # getting corner
                        corner = (int((box["x"] * 224) / width), int((box["y"] * 224) / height));
                        # getting dimensions
                        dims = (int((box["width"] * 224) / width), int((box["height"] * 224) / height));

                        # drawing bb
                        xray = cv2.rectangle(
                                xray,
                                (int(corner[0] + (dims[0])), int(corner[1] + dims[1])),
                                (int(corner[0]), int(corner[1])),
                                color = (1, 0, 0),
                                thickness = 2
                                );

                # saving image
                plt.imsave("/data/bb_imgs/{}.png".format(os.path.splitext(_file)[0]), xray, cmap = "gray", format = "png");


def draw_jet_bbs():

    # getting rid of write dir if it does exist
    if(os.path.isdir("/data/od_bbs")):
        shutil.rmtree("/data/od_bbs");

    # making dir
    os.makedirs("/data/od_bbs");

    # getting filenames in train
    filenames = [filename for filename in os.listdir("/data/kaggle_data/od_formatted/train")];

    # getting filename indices
    file_indices = random.sample(range(0, len(filenames)), 10);

    # opening labels
    image_bb = pandas.read_csv("/data/kaggle_data/labels/train_image_level.csv");

    # getting radians
    theta = numpy.pi / 2;

    # creating rotation matrix
    r_mat = numpy.array([
        [numpy.cos(theta), -1 * numpy.sin(theta)], 
        [numpy.sin(theta), numpy.cos(theta)]
    ]);

    # iterating over indices 
    for index in file_indices:

        # making path to file
        path = os.path.join("/data/kaggle_data/od_formatted/train", filenames[index]);

        # opening image
        img = numpy.asarray(Image.open(path).convert("RGB"));
        img = numpy.transpose(img).astype(numpy.uint8).copy();
        img = img[0];
        # getting kaggle id 
        kaggle_id = "{}_image".format(filenames[index].split("-")[-1].split(".")[0]);
        print(kaggle_id)
        # getting bounding boxes
        boxes = image_bb[image_bb["id"].str.contains(kaggle_id)].iloc[0].loc["boxes"];

        # reading in dicom
        xray_dicom = pydicom.filereader.dcmread(os.path.join(
            "/data/kaggle_data/raw_dicom/train", "{}.dcm".format(filenames[index].split(".")[0].replace("-", "/"))));

        # getting width amd height
        width = xray_dicom.pixel_array.shape[1];
        height = xray_dicom.pixel_array.shape[0];

        # filtering out Nans
        if(boxes == boxes):
            # getting json version of string
            j_boxes = json.loads(boxes.replace("'", "\""));

            # iterating over boxes
            for box in j_boxes:
                # getting mins
                mins = numpy.array([(box["x"] / width) - 0.0, (box["y"] / height) - 0.0]);

                # getting maxs
                maxs = numpy.array([((box["x"] + box["width"]) / width) - 0.0, ((box["y"] + box["height"]) / height) - 0.0]);

                # rotating
                r_mins = numpy.matmul(r_mat, mins);
                r_maxs = numpy.matmul(r_mat, maxs);

                # drawing bb
                img = cv2.rectangle(
                        img,
                        (int(r_maxs[0] * -1024), int(r_maxs[1] * 1024)),
                        (int(r_mins[0] * -1024), int(r_mins[1] * 1024)),
                        color = (1, 0, 0),
                        thickness = 2
                );


        # saving img
        save_img = Image.fromarray(img);
        save_img.save(os.path.join("/data/od_bbs", filenames[index]));
        save_img.close();

# draw_bounding_bs();
draw_jet_bbs();
