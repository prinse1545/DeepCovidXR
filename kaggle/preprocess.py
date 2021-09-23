# Filename: kaggle_data_preprocess.py
# Description: A script for processing
# the kaggle data into a model that can 
# be used by the ensemble.
# 
# 2021-07-08

# importing tools
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.transform import resize
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import importlib.util
import random
import argparse
import pydicom
import pandas
import numpy
import json
import shutil
import cv2
import os


def to_png(args):
    # Function: generate_images, a function that takes the kaggle dataset and
    # converts the dicoms to pngs. 

    # Warning: This function takes several hours to run.

    # Parameter(s):

    #     read_dir - the directory that should be read from
    #     write_dir - the directory that should be written to

    # Return Value(s):

    #     None

    def save_as_png(path_to_dcm, path_to_write, size = 512):

        # generating cropped images using image utils
        xray_dicom = pydicom.filereader.dcmread(path_to_dcm);
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
        xray = resize(xray, (size, size), anti_aliasing = True);
        
        # getting split path for filename
        path_split = path_to_dcm.split("/");

        # generating filename
        filename = "{}-{}-{}".format(path_split[-3], path_split[-2],
                os.path.splitext(path_split[-1])[0]);

        # writing image
        plt.imsave(os.path.join(path_to_write, "{}.png".format(filename)), xray, cmap = "gray", format = "png");

    # checking if path is a directory
    if(not os.path.isdir(args.read_dir)):
        # giving error message
        print("The path to the directory does not exist.");
        # exiting 
        return;
    if(not os.path.isdir(os.path.join(args.read_dir, "train")) or not os.path.isdir(os.path.join(args.read_dir, "test"))):
        # giving error message
        print("The given directory must contain a train and test directory");
        # exiting
        return;
    
    # if write directory exists delete it
    if(os.path.isdir(args.write_dir)):
        shutil.rmtree(args.write_dir);

    # creating write directory
    os.makedirs(os.path.join(args.write_dir, "train"));
    os.makedirs(os.path.join(args.write_dir, "test"));
    
    print("Converting training dicom files to png...\n");
    
    # initialzing training counter
    train_count = 0;

    # iterating over training data
    for subdir, dirs, files in os.walk(os.path.join(args.read_dir, "train")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(args.write_dir,
                "train"), 512);
            train_count = train_count + 1;
            print("Saved {} training images".format(train_count));

    
    print("Converting training dicom files to png...\n");

    # initializing testing counter
    test_count = 0;

    # iterating over testing data 
    for subdir, dirs, files in os.walk(os.path.join(args.read_dir, "test")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(args.write_dir,
                "test"), 512);
            test_count = test_count + 1;
            print("Saved {} testing images".format(test_count));



def create_classifier_filestruct(args):
    
    # checking if write directory exists and deleting if it does
    if(os.path.isdir(args.write_dir)):
        # deleting
        shutil.rmtree(args.write_dir);
    
    # keeping track of classes
    classes = ["no_pneumonia", "typical_appearance", "indeterminate_appearance", "atypical_appearance"];

    # keeping track of directories
    dirs = ["train", "valid", "test"];
    
    # creating directories
    for _class in classes:
        for _dir in dirs:
            # writing directory
            os.makedirs(os.path.join(args.write_dir, _dir, _class));

    # reading in labels
    labels = pandas.read_csv(args.labels);

    # initializing index dictionary where each index corresponds to a class
    class_index_dic = { 0:"no_pneumonia", 1:"typical_appearance",
            2:"indeterminate_appearance", 3:"atypical_appearance" };
    
    # getting list of filenames in data directory
    files = os.listdir(os.path.join(args.read_dir, "train"));

    # saving number of files
    n_files = len(files);

    # iterating over training data
    for index, file in enumerate(files):

        # getting full path
        full_path = os.path.join(args.read_dir, "train", file);

        # opening image
        img = Image.open(full_path);

        # resizing
        img = img.resize((224, 224));

        # getting study id
        study_id = file.split("-")[0];

        # getting csv id
        csv_id = "{}_study".format(study_id);
       
        # getting sample class
        sample_class = labels.iloc[labels.index[labels["id"] ==
            csv_id].tolist()[0]].to_numpy()[1:].argmax();
        
        # determining where to save
        save_location = None;

        if(index <= n_files / 5):
            save_location = "valid";
        elif(index <= (2 * n_files) / 5):
            save_location = "test";
        else:
            save_location = "train"

        # saving img
        img.save(os.path.join(args.write_dir, save_location,
            class_index_dic[sample_class], file));


def create_od_filestruct(args):
    
    def get_box_type(file, labels):
        # getting id name
        image_id = "{}_image".format(file.split("-")[-1].replace(".png", ""));

        # getting bounding boxes
        boxes = labels[labels["id"].str.contains(image_id)].iloc[0].loc["boxes"];

        # if boxes are not nan read in as json and calculate areas
        if(boxes == boxes):
            # initializing area
            areas = [];
            # loading boxes in json
            boxes = json.loads(boxes.replace("'", "\""));
            # iterating over boxes
            for box in boxes:
                # computing area
                area = box["width"] * box["height"];
                # appending
                areas.append(area);

            # returning max area
            max_area = max(areas);
        else:
            # max area is none if nothing
            max_area = None;

        # defining coarse or fine bool
        coarse_or_fine = None;

        # if nan for box we do it randomly
        if(max_area == None):
            coarse_or_fine = random.random() >= 0.5;
        else:
            coarse_or_fine = max_area >= 0.05;

        # returnin coarse or fine
        return "coarse" if(coarse_or_fine == True) else "fine";

    # checking that read and write dirs works
    if(not os.path.isdir(args.read_dir)):
        # printing error
        print("The read directory provided does not exist!");
        # exiting
        return;
    
    # checking if write dir exists and deleting if it does
    if(os.path.isdir(args.write_dir)):
        # delete it
        shutil.rmtree(args.write_dir);
    
    # defining write dirs
    write_dirs = ["valid", "train", "test"];

    # creating write dir
    for write_dir in write_dirs:
        # creating directories
        if(write_dir == "valid" or write_dir == "train"):
            os.makedirs(os.path.join(args.write_dir, write_dir, "fine"));
            os.makedirs(os.path.join(args.write_dir, write_dir, "coarse"));
        else:
            os.makedirs(os.path.join(args.write_dir, write_dir));

    # reading in labels as csv
    labels = pandas.read_csv(args.labels);

    # iterating over files in read dir
    for root, dir, files in os.walk(args.read_dir):
        if(root.split("/")[-1] == "train"):
            for file in files:

                # getting random number between zero and one
                num = random.random();
                # saving src name
                src = os.path.join(root, file);

                # opening image
                xray = ImageOps.grayscale(Image.open(src));

                # resizing
                xray = xray.resize((1024, 1024));

                # determining location to save
                location = None;

                # putting file in different places based on randomness
                if(num < 0.2):
                    # validation set
                    location = os.path.join("valid", get_box_type(file, labels));

                elif(num < 0.4):
                    # testing set
                    location = "test";
                else:
                    # training set
                    location = os.path.join("train", get_box_type(file, labels));

                # saving xray
                # xray.save(os.path.join(args.write_dir, location, file));

                # writing image
                plt.imsave(os.path.join(args.write_dir, location, file), numpy.asarray(xray), cmap = "bone", format = "png");
                
                # closing
                xray.close();
    # finished writing
    print("finished writing");


def preprocess_labels(args):

    # getting image labels
    i_labels = pandas.read_csv(args.labels);
    
    # defining image directory
    image_dir = os.path.join(args.read_dir, "train");

    # iterating over raw_dicoms
    for image_name in os.listdir(image_dir):

        # generating cropped images using image utils
        xray_dicom = pydicom.filereader.dcmread(os.path.join(args.dicom_dir, image_name.replace("-", "/").replace(".png", ".dcm")));

        # applying volume of interest look up table colors to get opacity in
        # accordance with dicom image format
        xray = apply_voi_lut(xray_dicom.pixel_array, xray_dicom);

        # getting width and height
        height, width = xray.shape;

        # creating image id
        image_id = "{}_image".format(image_name.split("-")[-1].replace(".png", ""));

        # getting bounding boxes
        boxes = i_labels[i_labels["id"].str.contains(image_id)].iloc[0].loc["boxes"];

        # checking if nan
        if(boxes == boxes):
            boxes = json.loads(boxes.replace("'", "\""));
        else:
            boxes = [];

        # iterating through boxes and normalizing
        for box in boxes:
            box["x"] = box["x"] / width;
            box["y"] = box["y"] / height;
            box["width"] = box["width"] / width;
            box["height"] = box["height"] / height;
        
        # updating boxes
        if(len(boxes) > 0):

            i_labels.loc[i_labels.index[i_labels["id"] == image_id].tolist()[0], "boxes"] = boxes;

        # saving as csv
        i_labels.to_csv(args.labels);


if(__name__ == "__main__"):
    # This is the main function of 
    # the preprocessing script.

    # initializing arparser
    parser = argparse.ArgumentParser();

    # adding arguments
    parser.add_argument("-r", "--read_dir", type = str, help = "Where the script reads from");
    parser.add_argument("-w", "--write_dir", type = str, help = "Where the script writes to");
    parser.add_argument("-d", "--dicom_dir", type = str, help = "The directory of the raw dicom files (needed when creating tf records)")
    parser.add_argument("-f", "--function", type = str, help = "What the script does. Either fetch_dataset, to_png, or create_filestruct");
    parser.add_argument("-l", "--labels", type = str, help = "Path to labels (provided by kaggle)");

    # getting args
    args = parser.parse_args();

    # creating functions dictionary
    functions = { 
            "create_od_filestruct": create_od_filestruct, 
            "to_png": to_png,
            "create_class_filestruct": create_classifier_filestruct,
            "preprocess_labels": preprocess_labels
    };

    # checking that there"s a function
    if(args.function == None):
        print("You must provide a function!");

    # exectuting function
    functions[args.function](args);

