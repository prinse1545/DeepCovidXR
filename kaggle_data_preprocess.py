# Filename: kaggle_data_preprocess.py
# Description: A script for processing
# the kaggle data into a model that can 
# be used by the ensemble.
# 
# 2021-07-08

# importing tools
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.transform import resize
from PIL import Image
import importlib.util
import tensorflow
import random
import argparse
import pydicom
import pandas
import numpy
import json
import shutil
import cv2
import os

# Importing object detectiuon from tf
spec = importlib.util.spec_from_file_location("dataset_util", "./object_detection/models/research/object_detection/utils/dataset_util.py");
dataset_util = importlib.util.module_from_spec(spec);
spec.loader.exec_module(dataset_util);

def to_png(read_dir, write_dir):
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
    if(not os.path.isdir(read_dir)):
        # giving error message
        print("The path to the directory does not exist.");
        # exiting 
        return;
    if(not os.path.isdir(os.path.join(read_dir, "train")) or not os.path.isdir(os.path.join(read_dir, "test"))):
        # giving error message
        print("The given directory must contain a train and test directory");
        # exiting
        return;
    
    # if write directory exists delete it
    if(os.path.isdir(write_dir)):
        shutil.rmtree(write_dir);

    # creating write directory
    os.makedirs(os.path.join(write_dir, "train"));
    os.makedirs(os.path.join(write_dir, "test"));
    
    print("Converting training dicom files to png...\n");
    
    # initialzing training counter
    train_count = 0;

    # iterating over training data
    for subdir, dirs, files in os.walk(os.path.join(read_dir, "train")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(write_dir,
                "train"), 512);
            train_count = train_count + 1;
            print("Saved {} training images".format(train_count));

    
    print("Converting training dicom files to png...\n");

    # initializing testing counter
    test_count = 0;

    # iterating over testing data 
    for subdir, dirs, files in os.walk(os.path.join(read_dir, "test")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(write_dir,
                "test"), 512);
            test_count = test_count + 1;
            print("Saved {} testing images".format(test_count));



def create_classifier_filestruct(labels_path, read_dir, write_dir):
    
    # checking if write directory exists and deleting if it does
    if(os.path.isdir(write_dir)):
        # deleting
        shutil.rmtree(write_dir);
    
    # keeping track of classes
    classes = ["no_pneumonia", "typical_appearance", "indeterminate_appearance", "atypical_appearance"];

    # keeping track of directories
    dirs = ["train", "valid", "test"];
    
    # creating directories
    for _class in classes:
        for _dir in dirs:
            # writing directory
            os.makedirs(os.path.join(write_dir, _dir, _class));

    # reading in labels
    labels = pandas.read_csv(labels_path);

    # initializing index dictionary where each index corresponds to a class
    class_index_dic = { 0:"no_pneumonia", 1:"typical_appearance",
            2:"indeterminate_appearance", 3:"atypical_appearance" };
    
    # getting list of filenames in data directory
    files = os.listdir(os.path.join(read_dir, "train"));

    # saving number of files
    n_files = len(files);

    # iterating over training data
    for index, file in enumerate(files):

        # getting full path
        full_path = os.path.join(read_dir, "train", file);

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
        img.save(os.path.join(write_dir, save_location,
            class_index_dic[sample_class], file));


def create_od_filestruct(args):
    
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

    # creating write dir
    os.makedirs(os.path.join(args.write_dir, "valid"));
    os.makedirs(os.path.join(args.write_dir, "train"));
    os.makedirs(os.path.join(args.write_dir, "test"));

    # iterating over files in read dir
    for root, dir, files in os.walk(args.read_dir):
        if(root.split("/")[-1] == "train"):
            for file in files:
                # getting random number between zero and one
                num = random.random();
                # saving src name
                print(root, file);
                src = os.path.join(root, file);
                # putting file in different places based on randomness
                if(num < 0.2):
                    # validation set
                    shutil.copyfile(src, os.path.join(args.write_dir, "valid", file));

                elif(num < 0.4):
                    # testing set
                    shutil.copyfile(src, os.path.join(args.write_dir, "test", file));
                else:
                    # training set
                    shutil.copyfile(src, os.path.join(args.write_dir, "train", file));

    # finished writing
    print("finished writing");

def create_tf_records(args):

    # checking for read dir
    if(not args.read_dir):
        # printing err
        print("The read directory is not provided");
        # exit
        return;

    if(not args.labels):
        # printing err
        print("The labels are not provided");
        # exit
        return;

    # reading in csv
    xray_bb = pandas.read_csv(args.labels);

    # iterating over read dir
    for root, dir, files in os.walk(args.read_dir):
        # creating writer
        writer = None;
        direc = root.split("/")[-1];

        if(direc != "valid" and direc != "train" and direc != "test"):
            continue;
        
        writer = tensorflow.io.TFRecordWriter(os.path.join(args.write_dir, "{}-record.tfrecord".format(direc)));

        for file in files:
            # getting kaggle id 
            kaggle_id = "{}_image".format(os.path.splitext(file.split("-")[-1])[0]);
            # getting bounding boxes
            boxes = xray_bb[xray_bb["id"].str.contains(kaggle_id)].iloc[0].loc["boxes"];

            # getting boxes in json not string
            if(boxes == boxes):
                boxes = json.loads(boxes.replace("'", "\""));
            else:
                boxes = [];

            # getting dicom to get dimensions
            xray = pydicom.filereader.dcmread(os.path.join(args.dicom_dir, file.replace("-", "/").replace(".png", ".dcm")));
            
            # getting height and width
            width = xray.pixel_array.shape[1];
            height = xray.pixel_array.shape[0];
            
            # reading image as png to encode for tf
            xray = cv2.imread(os.path.join(root, file));
            _, xray_encoded = cv2.imencode(".png", xray);

            # creating boxes for tf records
            xmins = [];
            xmaxs = [];
            ymins = [];
            ymaxs = [];
            classes_text = [];
            classes_num = [];

            for box in boxes:
                # taking care of classes
                classes_text.append(b"opacity") # we are only detecting opacities
                classes_num.append(1);
                # taking care of values
                xmins.append((box["x"] + box["width"]) / width);
                xmaxs.append(box["x"] / width);
                ymins.append((box["y"] - box["height"]) / height);
                ymaxs.append(box["y"] / height);

            # making features
            feats = {
                "image/height": dataset_util.int64_feature(224),
                "image/width": dataset_util.int64_feature(224),
                "image/filename": dataset_util.bytes_feature(str.encode(file)),
                "image/source_id": dataset_util.bytes_feature(str.encode(file)),
                "image/encoded": dataset_util.bytes_feature(xray_encoded.tobytes()),
                "image/format": dataset_util.bytes_feature(b"png"),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
                "image/object/class/label": dataset_util.int64_list_feature(classes_num)
            };

            # making tf example
            tf_example = tensorflow.train.Example(features = tensorflow.train.Features(feature = feats));
            # writing example
            writer.write(tf_example.SerializeToString());

        # closing writer
        if(writer != None):
            writer.close();


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
    parser.add_argument("-l", "--labels", type = str, help = "Where the labels provided by kaggle live (should be a csv)");

    # getting args
    args = parser.parse_args();

    # creating functions dictionary
    functions = { "create_od_filestruct": create_od_filestruct, "create_tf_records": create_tf_records };

    # checking that there"s a function
    if(args.function == None):
        print("You must provide a function!");

    # exectuting function
    functions[args.function](args);

