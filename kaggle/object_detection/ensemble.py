# Filename: ensemble.py
# Description: This file implements an ensemble of
# faster rcnn models
# 
# 2021-09-09

# importing tools
import torch
from ensemble_boxes import weighted_boxes_fusion, non_maximum_weighted

class od_ensemble():
    # initializing ensemble
    def __init__(self, models, create_model):
        # saving model names
        self.model_names = models;
        self.create_model = create_model;

    def to(self, device):
        # enabling GPU if available
        self.device = device;

        # initialzing loaded models
        loaded_models = [];

        # loading models
        for model_info in self.model_names:
            # loading
            model = self.create_model(model_info[0], model_info[1], model_info[2]);
            
            # sending to device
            model.to(self.device);

            # appending
            loaded_models.append(model);

        # saving ensemble
        self.ensemble = loaded_models;

        # returning self
        return self;

    def eval(self):

        # eval for all models
        for model in self.ensemble:
            model.eval();

    def __join_bounding_boxes(self, pred_arr):

        # initializing final pred array
        interm_preds = [];
        final_boxes = [];
        final_scores = [];
        final_labels = [];
        
        for preds in pred_arr:
            # initializing empty box list
            boxes = [];
            scores = [];
            labels = [];

            # iterating over predictions
            for pred in preds:
                # init norm boxes
                norm_boxes = [];

                # normalizing predictions
                for box in pred["boxes"].cpu().detach().numpy().tolist():
                    norm_boxes.append([coord / 1024 for coord in box]);

                # getting info
                boxes.append(norm_boxes);
                scores.append(pred["scores"].cpu().detach().numpy().tolist());
                labels.append(pred["labels"].cpu().detach().numpy().tolist());

            # appending fused bbs
            final_boxes.append(boxes);
            final_scores.append(scores);
            final_labels.append(labels);

        # print("BOX\n", final_boxes[0], "\nspace\n",  final_boxes[1], "\nBOX END\n")
        for boxes, scores, labels in zip(zip(*final_boxes), zip(*final_scores), zip(*final_labels)):
            # appending final boxes

            # print(list(labels))
            interm_preds.append(non_maximum_weighted(list(boxes), list(scores), list(labels),
                  weights = [2, 1], iou_thr = 0.3, skip_box_thr = 0.4));

        # initialzing final preds
        final_preds = [];

        for pred in interm_preds:
            boxes = [box * 1024 for box in pred[0]];
            final_preds.append({ "boxes": torch.as_tensor(boxes), "scores": torch.as_tensor(pred[1]), "labels": torch.as_tensor(pred[2]) });

        # returning final preds
        return final_preds;

    def __call__(self, images):


        # initializing predictions
        predictions = [];
    
        # getting predictions
        for model in self.ensemble:
    
            # predicting
            predictions.append(model(images));
    
        # joining bounding boxes
        final_preds = self.__join_bounding_boxes(predictions);
    
        # returning final predictions
        return final_preds;

