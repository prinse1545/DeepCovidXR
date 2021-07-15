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
import shutil
import pydicom
import pandas
import random
import numpy
import cv2
import json
import os

def draw_bounding_bs():
    # This function takes 100 random images and 
    # draws their bounding boxes

    # dicom path
    dicom_path = "/data/kaggle_data";

    # getting filenames
    filenames = [];

    for _, _, files in os.walk(os.path.join(dicom_path, "train")):
        filenames.extend(files);

    # getting random numbers for filenames
    # file_indices = random.sample(range(0, len(filenames)), 100);
    file_indices = [5225, 435, 1626, 4929, 3643, 1560, 1954, 4394, 3982, 2740, 3255, 6037, 2078, 2331, 844, 2408, 4925, 5654, 2885, 2666, 5786, 783, 126, 2362, 1675, 4228, 3586, 2865, 5621, 1972, 3815, 6256, 3863, 1125, 647, 3298, 1097, 5652, 4602, 3581, 3501, 3187, 344, 1117, 1144, 4585, 4981, 1804, 4740, 4550, 59, 3777, 4821, 6238, 3292, 4723, 1342, 5603, 5154, 6152, 4473, 841, 2317, 2656, 2492, 4018, 434, 4236, 6158, 4949, 1630, 829, 5606, 2071, 4620, 4969, 4893, 4625, 3638, 2804, 4244, 3938, 1430, 1376, 2704, 4995, 838, 5008, 5871, 1558, 5898, 1356, 1321, 1624, 2252, 3001, 850, 321, 4809, 1864]

    # getting filenames to draw
    draw_filenames = [filenames[indx] for indx in file_indices];

    # getting training image bounding box data
    image_bb = pandas.read_csv("/data/kaggle_data/train_image_level.csv");

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
                        # getting center
                        center = (int((box["x"] * 224) / width), int((box["y"] * 224) / height));
                        # getting dimensions
                        dims = (int((box["width"] * 224) / width), int((box["height"] * 224) / height));

                        # drawing bb
                        xray = cv2.rectangle(
                                xray,
                                (int(center[0]), int(center[1] - (dims[1]))),
                                (int(center[0] + (dims[0])), int(center[1])),
                                color = (1, 0, 0),
                                thickness = 2
                                );

                # saving image
                plt.imsave("/data/bb_imgs/{}.png".format(os.path.splitext(_file)[0]), xray, cmap = "gray", format = "png");

draw_bounding_bs();
