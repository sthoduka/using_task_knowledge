import os
import sys
import random
import glob
import json
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools

import decord

import torchvision
from torchvision.transforms import v2 as transforms
import torch
import torch.nn.functional as F

import pyarrow.parquet as pq

from datasets.utils import (
        variable_fps_frame_selection,
        constant_fps_frame_selection,
        non_action_aligned_variable_fps_frame_selection
)


label_encoder = {"nominal": 0, "deconstruction": 1, "open": 2}
action_classes = {"ApproachGrasp": 0, "MoveDestination": 1, "Wait": 2, "ApproachPlace": 3, "MoveSource": 4}


def load_data(data_root, trials, hparams, training):
    data_video_path = {}
    labels = {}
    data_frame_sequences = {}
    data_boxes = {}
    data_feat = {}
    data_feat_length = {}
    data_action = {}
    data_crop_boxes = {}
    data_defect_segments = {}
    # used for test time augmentation; trials are referred to multiple times
    data_trial_action = []
    data_selected_trials = []


    # robot actions (extracted using trained action segmentation model)
    detected_segments_df = pq.read_table(os.path.join(data_root, 'detected_segments.parquet'))
    detected_segments_df = detected_segments_df.to_pandas()
    detected_segments_df["segments"] = detected_segments_df["segments"].apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    detected_segments_df.set_index("trial", inplace=True)

    # robot actions (manually annotated)
    annotated_segments_df = pq.read_table(os.path.join(data_root, 'annotated_segments.parquet'))
    annotated_segments_df = annotated_segments_df.to_pandas()
    annotated_segments_df["segments"] = annotated_segments_df["segments"].apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    annotated_segments_df.set_index("trial", inplace=True)

    # combined boxes for src box, arm, and destination box
    crop_boxes_df = pq.read_table(os.path.join(data_root, 'crop_boxes.parquet'))
    crop_boxes_df = crop_boxes_df.to_pandas()
    crop_boxes_df["boxes"] = crop_boxes_df.apply(lambda row: np.frombuffer(row["boxes"], dtype=np.float64).reshape(row["shape"]), axis=1)
    crop_boxes_df.set_index("trial", inplace=True)

    # frame-wise defect labels
    defect_segments_df = pq.read_table(os.path.join(data_root, 'defect_segments.parquet'))
    defect_segments_df = defect_segments_df.to_pandas()
    defect_segments_df["defect_segments"] = defect_segments_df["defect_segments"].apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    defect_segments_df.set_index("trial", inplace=True)

    for idx, trial in enumerate(trials):
        trial_name = trial

        # prefer annotated segments over detected segments
        if trial_name in annotated_segments_df.index:
            actions = annotated_segments_df.loc[trial_name].segments.copy()
        elif trial_name in detected_segments_df.index:
            actions = detected_segments_df.loc[trial_name].segments.copy()

        if hparams.selected_action != '':
            selected_action_frames = np.where(actions == action_classes[hparams.selected_action])[0]
            if len(selected_action_frames) == 0:
                continue
        data_video_path[trial_name] = os.path.join(data_root, 'videos', trial_name + '.mp4')

        # prefer relabeled annotations over original
        rlbl_path = os.path.join(data_root, 'relabeled_annotations', trial_name + '.json')
        if os.path.exists(rlbl_path):
            with open(os.path.join(data_root, "relabeled_annotations",  trial_name + '.json'), 'r') as fp:
                annotation = json.load(fp)
        else:
            with open(os.path.join(data_root, "annotations",  trial_name + '.json'), 'r') as fp:
                annotation = json.load(fp)
        labels[trial_name] = annotation["label"]
        data_action[trial_name] = actions

        crop_boxes = crop_boxes_df.loc[trial_name].boxes.copy()
        if hparams.action_crop:
            crop_boxes = crop_boxes[:, 1, :]
            crop_boxes = replace_with_largest_box_actionwise(crop_boxes, actions)
        data_crop_boxes[trial_name] = crop_boxes

        if trial_name in defect_segments_df.index:
            defect_segments = defect_segments_df.loc[trial_name].defect_segments.copy()
        else:
            defect_segments = np.zeros_like(actions)
        data_defect_segments[trial_name] = defect_segments

        if hparams.action_subset_frame_selection:
            move_destination_frames = np.where(actions == action_classes["MoveDestination"])[0]
            transition_idx = np.where(np.abs(np.diff(move_destination_frames)) > 1)[0]
            if len(transition_idx) > 0:
                move_destination_frames = move_destination_frames[:transition_idx[0]+1]

            wait_frames = np.where(actions == action_classes["Wait"])[0]
            transition_idx = np.where(np.abs(np.diff(wait_frames)) > 1)[0]
            if len(transition_idx) > 0:
                wait_frames = wait_frames[:transition_idx[0]+1]

            approach_place_frames = np.where(actions == action_classes["ApproachPlace"])[0]
            transition_idx = np.where(np.abs(np.diff(approach_place_frames)) > 1)[0]
            if len(transition_idx) > 0:
                approach_place_frames = approach_place_frames[:transition_idx[0]+1]

            move_source_frames = np.where(actions == action_classes["MoveSource"])[0]
            transition_idx = np.where(np.abs(np.diff(move_source_frames)) > 1)[0]
            if len(transition_idx) > 0:
                move_source_frames = move_source_frames[:transition_idx[0]+1]

            selected_frames = np.concatenate((move_destination_frames, wait_frames, approach_place_frames))

            data_frame_sequences[trial_name] = selected_frames
        elif 'img_pair' in hparams.dataset:
            move_destination_frames = np.where(actions == action_classes["MoveDestination"])[0]
            wait_frames = np.where(actions == action_classes["Wait"])[0]
            approach_place_frames = np.where(actions == action_classes["ApproachPlace"])[0]
            selected_frames = np.concatenate((move_destination_frames, wait_frames, approach_place_frames))
            selected_frames.sort()
            data_frame_sequences[trial_name] = selected_frames
        else:
            data_frame_sequences[trial_name] = []


        if hparams.action_aligned_fps_aug and not training: # we need to add multiple samples per trial
            action_types = ["MoveDestination", "Wait", "ApproachPlace"]
            for at in action_types:
                selected_action_frames = np.where(actions == action_classes[at])[0]
                transition_idx = np.where(np.abs(np.diff(selected_action_frames)) > 1)[0]
                if len(transition_idx) > 0:
                    selected_action_frames = selected_action_frames[:transition_idx[0]+1]
                if len(selected_action_frames) > 10:
                    data_trial_action.append(action_classes[at])
                    data_selected_trials.append(trial_name)
            # always add normal frame rate
            data_trial_action.append(0)
            data_selected_trials.append(trial_name)
        else:
            data_selected_trials.append(trial_name)


    data = {}
    data['video'] = data_video_path
    data['label'] = labels
    data['frame_seq'] = data_frame_sequences
    data['action'] = data_action
    data['crop_box'] = data_crop_boxes
    data['defect_segments'] = data_defect_segments
    data['trial'] = data_selected_trials
    data['trial_action'] = data_trial_action
    return data

def load_data_actionwise(data_root, trials, hparams, training):

    detected_segments_df = pq.read_table(os.path.join(data_root, 'detected_segments.parquet'))
    detected_segments_df = detected_segments_df.to_pandas()
    detected_segments_df["segments"] = detected_segments_df["segments"].apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    detected_segments_df.set_index("trial", inplace=True)

    annotated_segments_df = pq.read_table(os.path.join(data_root, 'annotated_segments.parquet'))
    annotated_segments_df = annotated_segments_df.to_pandas()
    annotated_segments_df["segments"] = annotated_segments_df["segments"].apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    annotated_segments_df.set_index("trial", inplace=True)

    crop_boxes_df = pq.read_table(os.path.join(data_root, 'crop_boxes.parquet'))
    crop_boxes_df = crop_boxes_df.to_pandas()
    crop_boxes_df["boxes"] = crop_boxes_df.apply(lambda row: np.frombuffer(row["boxes"], dtype=np.float64).reshape(row["shape"]), axis=1)
    crop_boxes_df.set_index("trial", inplace=True)

    # frame-wise defect labels
    defect_segments_df = pq.read_table(os.path.join(data_root, 'defect_segments.parquet'))
    defect_segments_df = defect_segments_df.to_pandas()
    defect_segments_df["defect_segments"] = defect_segments_df["defect_segments"].apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    defect_segments_df.set_index("trial", inplace=True)

    data_actions = {}
    data_video_path = {}
    labels = {}
    data_boxes = {}
    data_action = {}
    data_crop_boxes = {}
    data_defect_segments = {}

    data_selected_trials = []
    data_trial_action = []

    for idx, trial in enumerate(trials):
        trial_name = trial

        # prefer annotated segments over detected segments
        if trial_name in annotated_segments_df.index:
            actions = annotated_segments_df.loc[trial_name].segments.copy()
        elif trial_name in detected_segments_df.index:
            actions = detected_segments_df.loc[trial_name].segments.copy()
        data_actions[trial_name] = actions

        data_video_path[trial_name] = os.path.join(data_root, 'videos', trial_name + '.mp4')

        # prefer relabeled annotations over original
        rlbl_path = os.path.join(data_root, 'relabeled_annotations', trial_name + '.json')
        if os.path.exists(rlbl_path):
            with open(os.path.join(data_root, "relabeled_annotations",  trial_name + '.json'), 'r') as fp:
                annotation = json.load(fp)
        else:
            with open(os.path.join(data_root, "annotations",  trial_name + '.json'), 'r') as fp:
                annotation = json.load(fp)
        labels[trial_name] = annotation["label"]

        crop_boxes = crop_boxes_df.loc[trial_name].boxes.copy()
        if hparams.action_crop:
            crop_boxes = crop_boxes[:, 1, :]
            crop_boxes = replace_with_largest_box_actionwise(crop_boxes, actions)
        data_crop_boxes[trial_name] = crop_boxes

        if trial_name in defect_segments_df.index:
            defect_segments = defect_segments_df.loc[trial_name].defect_segments.copy()
        else:
            defect_segments = np.zeros_like(actions)
        data_defect_segments[trial_name] = defect_segments

        action_types = ["MoveDestination", "Wait", "ApproachPlace"]
        for at in action_types:
            selected_action_frames = np.where(actions == action_classes[at])[0]
            transition_idx = np.where(np.abs(np.diff(selected_action_frames)) > 1)[0]
            if len(transition_idx) > 0:
                selected_action_frames = selected_action_frames[:transition_idx[0]+1]
            if len(selected_action_frames) > 10:
                data_trial_action.append(at)
                data_selected_trials.append(trial_name)

    data = {}
    data['video'] = data_video_path
    data['label'] = labels
    data['crop_box'] = data_crop_boxes
    data['action'] = data_actions
    data['defect_segments'] = data_defect_segments
    data['trial'] = data_selected_trials
    data['trial_action'] = data_trial_action
    return data

def replace_with_largest_box_actionwise(boxes, actions, height=560, left_edge=280):
    '''
    replace boxes with the maximal bounding box for the two regions
    i.e. find the largest bounding box for the pick-up region (left) and 
    the place region (right) based on the boxes corresponding to the actions
    performed in those regions
    '''
    # find valid boxes
    valid_box_idx = np.where(np.min(boxes, axis=1) >= 0)
    valid_boxes = boxes[valid_box_idx]
    valid_actions = actions[valid_box_idx]
    # boxes for approach, movedest, movesrc
    left_box_idx = np.where((valid_actions == 0) | (valid_actions == 1) | (valid_actions == 4))
    # boxes for wait, place
    right_box_idx = np.where((valid_actions == 2) | (valid_actions == 3))
    # convert to xyxy
    if left_box_idx[0].shape[0] != 0:
        left_boxes = np.apply_along_axis(get_xyxy, 1, valid_boxes[left_box_idx])
        # find min/max coordinates
        left_min, left_max = np.min(left_boxes, axis=0), np.max(left_boxes, axis=0)
        # left box with fixed height and left edge
        left_box_width = left_max[2] - left_min[0]
        left_box_mid_x = int(left_min[0] + (left_box_width / 2.0))
        left_box = np.array([left_box_mid_x, left_edge, left_box_width, height])
        # replace the boxes based on the action
        left_box_idx = np.where((actions == 0) | (actions == 1) | (actions == 4))
        boxes[left_box_idx] = left_box
    if right_box_idx[0].shape[0] != 0:
        right_boxes = np.apply_along_axis(get_xyxy, 1, valid_boxes[right_box_idx])
        # find min/max coordinates
        right_min, right_max = np.min(right_boxes, axis=0), np.max(right_boxes, axis=0)
        # right box with height equal to width
        right_box_width = right_max[2] - right_min[0]
        right_box_height = right_box_width
        right_box_mid_x = int(right_min[0] + (right_box_width / 2.0))
        right_box_mid_y = int(right_min[1] + (right_box_height / 2.0))
        right_box = np.array([right_box_mid_x, right_box_mid_y, right_box_width, right_box_height])
        # replace the boxes based on the action
        right_box_idx = np.where((actions == 2) | (actions == 3))
        boxes[right_box_idx] = right_box
    return boxes


def get_xyxy(box, img_width=1280, img_height=560):
    '''
    get start and end point within the image, corresponding to the box
    '''
    sp = [int(box[0] - (box[2] / 2.0)), int(box[1] - (box[3] / 2.0))]
    ep = [int(box[0] + (box[2] / 2.0)), int(box[1] + (box[3] / 2.0))]
    sp[0] = max(sp[0], 0)
    sp[1] = max(sp[1], 0)
    ep[0] = min(ep[0], img_width)
    ep[1] = min(ep[1], img_height)
    return np.array([sp[0], sp[1], ep[0], ep[1]])


def get_img_pair(video_path, frame_seq, hparams, scene_crop_boxes=None, training=True, data=None, index=None, trial=None):
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path)
    label = data['label'][trial]
    robot_actions = data['action'][trial]
    if hparams.selected_action != '' or hparams.actions_separately:
        if hparams.selected_action != '':
            selected_action = hparams.selected_action
        else:
            selected_action = data['trial_action'][index]
        action_index = action_classes[selected_action]
        selected_action_frames = np.where(robot_actions == action_index)[0]
        transition_idx = np.where(np.abs(np.diff(selected_action_frames)) > 1)[0]
        if len(transition_idx) > 0:
            selected_action_frames = selected_action_frames[:transition_idx[0]+1]
    start_frame, end_frame = selected_action_frames[0], selected_action_frames[-1]
    if training:
        if start_frame + 10 < end_frame:
            start_frame = start_frame + np.random.randint(0, 10)
        if end_frame - 10 > start_frame:
            end_frame = end_frame - np.random.randint(0, 10)
    selected_frames = [start_frame, end_frame]
    vid = vr.get_batch(selected_frames)
    if scene_crop_boxes is not None:
        scene_boxes = scene_crop_boxes[selected_frames]
        cropped_frames = []
        for idx, frame in enumerate(vid):
            xyxy = get_xyxy(scene_boxes[idx])
            cropped_frame = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
            cropped_frame = cropped_frame.permute(2, 0, 1) # C x H x W

            cropped_frame = transforms.functional.resize(cropped_frame, (560, 750)).permute(1, 2, 0)
            cropped_frames.append(cropped_frame)
        vid = torch.stack(cropped_frames)
    else:
        vid = vid[:, :, 100:850, :]
    vid = vid.permute(0, 3, 1, 2) # T x C x H x W
    selected_frames = list(range(start_frame, end_frame))
    return vid, selected_frames

def get_video(video_path, frame_seq, hparams, scene_crop_boxes=None, training=True, data=None, index=None, trial=None):
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path)
    num_frames_to_sample = 32
    if len(frame_seq):
        # we just have one array of frame ids
        if isinstance(frame_seq, np.ndarray):
            if training:
                start_frame = np.random.randint(0, 5)
            else:
                start_frame = 0
            selected_frames = frame_seq[np.round(np.linspace(start_frame, len(frame_seq)-1, num_frames_to_sample)).astype(int)]
    elif hparams.selected_action != '' or hparams.actions_separately:
        actions = data['action'][trial]
        if hparams.selected_action != '':
            selected_action = hparams.selected_action
        else:
            selected_action = data['trial_action'][index]

        if selected_action == "MoveDestination":
            move_destination_frames = np.where(actions == action_classes["MoveDestination"])[0]
            transition_idx = np.where(np.abs(np.diff(move_destination_frames)) > 1)[0]
            if len(transition_idx) > 0:
                move_destination_frames = move_destination_frames[:transition_idx[0]+1]
            action_frames = move_destination_frames

        if selected_action == "Wait":
            wait_frames = np.where(actions == action_classes["Wait"])[0]
            transition_idx = np.where(np.abs(np.diff(wait_frames)) > 1)[0]
            if len(transition_idx) > 0:
                wait_frames = wait_frames[:transition_idx[0]+1]
            action_frames = wait_frames

        if selected_action == "ApproachPlace":
            approach_place_frames = np.where(actions == action_classes["ApproachPlace"])[0]
            transition_idx = np.where(np.abs(np.diff(approach_place_frames)) > 1)[0]
            if len(transition_idx) > 0:
                approach_place_frames = approach_place_frames[:transition_idx[0]+1]
            action_frames = approach_place_frames

        if training and len(action_frames) > 10:
            start_frame = np.random.randint(0, 5)
        else:
            start_frame = 0
        selected_frames = action_frames[np.round(np.linspace(start_frame, len(action_frames)-1, num_frames_to_sample)).astype(int)]
    elif hparams.action_aligned_fps_aug:
        actions = data['action'][trial]
        low_fps_action_counts = {1: int(0.25 * num_frames_to_sample), 2: int(0.125 * num_frames_to_sample), 3: int(0.125 * num_frames_to_sample)}
        unique_action_ids = list(action_classes.values())
        if training:
            while True:
                selected_action = np.random.randint(0, 4)
                if selected_action == 0:
                    break
                if len(np.where(actions == selected_action)[0]) > 10:
                    break
        else:
            selected_action = data['trial_action'][index]

        if selected_action == 0:
            selected_frames = constant_fps_frame_selection(
                    actions,
                    low_fps_action_counts,
                    unique_action_ids,
                    training=training
            )
        else:
            high_fps_action = selected_action
            del low_fps_action_counts[high_fps_action]
            selected_frames = variable_fps_frame_selection(
                    actions,
                    high_fps_action,
                    low_fps_action_counts,
                    unique_action_ids,
                    training=training
            )
    else: # baseline, uniform sampling from full video
        if training:
            start_frame = np.random.randint(0, 5)
        else:
            start_frame = 0
        if hparams.non_action_aligned_fps_aug:
            if np.random.randint(0, 4) == 0:
                selected_frames = np.round(np.linspace(start_frame, len(vr)-1, num_frames_to_sample)).astype(int)
            else:
                selected_frames = non_action_aligned_variable_fps_frame_selection(len(vr))
        else:
            selected_frames = np.round(np.linspace(start_frame, len(vr)-1, num_frames_to_sample)).astype(int)
    vid = vr.get_batch(selected_frames)
    blank_images_idx = np.where(selected_frames == -1)[0]
    if len(blank_images_idx) > 0:
        vid[blank_images_idx] = 0
    if scene_crop_boxes is not None:
        # crop based on action
        scene_boxes = scene_crop_boxes[selected_frames]
        cropped_frames = []
        for idx, frame in enumerate(vid):
            xyxy = get_xyxy(scene_boxes[idx])
            cropped_frame = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
            cropped_frame = cropped_frame.permute(2, 0, 1) # C x H x W

            cropped_frame = transforms.functional.resize(cropped_frame, (560, 750)).permute(1, 2, 0)
            cropped_frames.append(cropped_frame)
        vid = torch.stack(cropped_frames)
    else:
        # standard cropping applied by baseline
        vid = vid[:, :, 100:850, :]
    vid = vid.permute(0, 3, 1, 2) # T x C x H x W
    return vid, selected_frames

def get_updated_label(defect_segments, selected_action):
    '''
    update the label of the clip based on whether the defect label
    changes within the clip
    1. changes twice: nominal->open->deconstruction
    2. changes once: nominal->deconstruction or open->deconstruction or nominal->open
    3. no change: if deconstruction, the current state of the object would be 'open' if the robot is not in MoveDestination
                  and 'deconstruction' if the robot is still in MoveDestination
                  for nominal/open, the current state remains the same

    '''
    transition_indices = np.where(defect_segments[:-1] != defect_segments[1:])[0]
    if len(transition_indices) > 1: #  there have been two transitions
        one_hot_lbl = torch.tensor([1., 0.]) # decons
    elif len(transition_indices) > 0: #  there has been one transition
        if 1 in defect_segments:
            one_hot_lbl = torch.tensor([1., 0.]) # decons
        elif 2 in defect_segments:
            one_hot_lbl = torch.tensor([0., 1.]) # open
    else: # no transition
        if 0 in defect_segments:
            one_hot_lbl = torch.tensor([0., 0.]) # nominal
        elif 1 in defect_segments:
            if selected_action != 'MoveDestination':
                one_hot_lbl = torch.tensor([0., 1.]) # if deconstruction has already happened, the object is "open"
            else:
                one_hot_lbl = torch.tensor([1., 0.]) # we're still in MoveDest, so it's a deconstruction
        elif 2 in defect_segments:
            one_hot_lbl = torch.tensor([0., 1.]) # if it was already open, it's still open
    return one_hot_lbl

class ArmbenchVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, json_file, hparams, vid_transform=None, training=True):
        self.data_root = data_root
        with open(json_file, 'r') as fp:
            self.trials = json.load(fp)
        self.data = load_data(data_root, self.trials, hparams, training)
        if len(self.data['trial']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)
        self.training = training
        self.vid_transform = vid_transform
        self.hparams = hparams

    def __getitem__(self, index):
        trial_name = self.data['trial'][index]

        label = self.data['label'][trial_name]
        video_path = self.data['video'][trial_name]
        frame_seq = self.data['frame_seq'][trial_name]

        if self.hparams.action_crop:
            vid, selected_frames = get_video(
                    video_path,
                    frame_seq,
                    self.hparams,
                    scene_crop_boxes=self.data['crop_box'][trial_name],
                    training=self.training,
                    data=self.data,
                    index=index,
                    trial=trial_name
            )
        else:
            vid, selected_frames = get_video(
                    video_path,
                    frame_seq,
                    self.hparams,
                    training=self.training,
                    data=self.data,
                    index=index,
                    trial=trial_name
            )

        if self.vid_transform is not None:
            vid = self.vid_transform(vid)

        vid = vid.permute(1, 0, 2, 3) # C X T x H x W
        lbl = label_encoder[label]
        if lbl == 0: # nominal
            one_hot_lbl = torch.tensor([0., 0.])
        elif lbl == 1: # deconstruction
            one_hot_lbl = torch.tensor([1., 0.])
        elif lbl == 2: # open
            one_hot_lbl = torch.tensor([0., 1.])

        if self.hparams.selected_action != '':
            one_hot_lbl = get_updated_label(self.data['defect_segments'][trial_name][selected_frames], self.hparams.selected_action)

        robot_actions = self.data['action'][trial_name][selected_frames]

        return vid, robot_actions, one_hot_lbl, video_path

    def __len__(self):
        return len(self.data['trial'])



class ArmbenchVideoActionWiseDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, json_file, hparams, vid_transform=None, training=True):
        self.data_root = data_root
        with open(json_file, 'r') as fp:
            self.trials = json.load(fp)

        self.data = load_data_actionwise(data_root, self.trials, hparams, training)
        if len(self.data['trial']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)
        self.training = training
        self.vid_transform = vid_transform
        self.hparams = hparams

    def __getitem__(self, index):
        trial_name = self.data['trial'][index]
        selected_action = self.data['trial_action'][index]

        label = self.data['label'][trial_name]
        video_path = self.data['video'][trial_name]
        frame_seq = []
        if self.hparams.action_crop:
            vid, selected_frames = get_video(
                    video_path,
                    frame_seq,
                    self.hparams,
                    scene_crop_boxes=self.data['crop_box'][trial_name],
                    training=self.training,
                    data=self.data,
                    trial=trial_name,
                    index=index
            )
        else:
            vid, selected_frames = get_video(
                    video_path,
                    frame_seq,
                    self.hparams,
                    training=self.training,
                    data=self.data,
                    trial=trial_name,
                    index=index
            )

        if self.vid_transform is not None:
            vid = self.vid_transform(vid)
        vid = vid.permute(1, 0, 2, 3) # C X T x H x W
        lbl = label_encoder[label]
        if lbl == 0:
            one_hot_lbl = torch.tensor([0., 0.])
        elif lbl == 1:
            one_hot_lbl = torch.tensor([1., 0.])
        elif lbl == 2:
            one_hot_lbl = torch.tensor([0., 1.])

        one_hot_lbl = get_updated_label(self.data['defect_segments'][trial_name][selected_frames], self.hparams.selected_action)

        robot_actions = self.data['action'][trial_name][selected_frames]
        robot_actions = robot_actions[::2]

        return vid, robot_actions, one_hot_lbl, video_path

    def __len__(self):
        return len(self.data['trial'])


class ArmbenchImgPairActionWiseDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, json_file, hparams, vid_transform=None, training=True):
        self.data_root = data_root
        with open(json_file, 'r') as fp:
            self.trials = json.load(fp)
        self.data = load_data_actionwise(data_root, self.trials, hparams, training)
        if len(self.data['trial']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)
        self.training = training
        self.vid_transform = vid_transform
        self.hparams = hparams

    def __getitem__(self, index):
        trial_name = self.data['trial'][index]
        selected_action = self.data['trial_action'][index]

        label = self.data['label'][trial_name]
        video_path = self.data['video'][trial_name]

        frame_seq = []
        if self.hparams.action_crop:
            vid, selected_frames = get_img_pair(
                    video_path,
                    frame_seq,
                    self.hparams,
                    scene_crop_boxes=self.data['crop_box'][trial_name],
                    training=self.training,
                    data=self.data,
                    index=index,
                    trial=trial_name
            )
        else:
            vid, selected_frames = get_img_pair(
                    video_path,
                    frame_seq,
                    self.hparams,
                    training=self.training,
                    data=self.data,
                    index=index,
                    trial=trial_name
            )

        if self.vid_transform is not None:
            vid = self.vid_transform(vid) # T x C x H x W
        lbl = label_encoder[label]

        one_hot_lbl = get_updated_label(self.data['defect_segments'][trial_name][selected_frames], selected_action)
        robot_actions = self.data['action'][trial_name][selected_frames][0]

        return vid, robot_actions, one_hot_lbl, video_path

    def __len__(self):
        return len(self.data['trial'])


def get_armbench_dataset(hparams, dataset_type='train'):
    if hparams.dataset == "armbench_video":
        crop_size = 650
        if dataset_type == 'train':
            vid_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(crop_size, pad_if_needed=True),
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
            )
        else:
            vid_transform = transforms.Compose(
                    [
                        transforms.CenterCrop(crop_size),
                        transforms.Resize(224),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
            )
        trials = {'train': hparams.training_trials, 'val': hparams.val_trials, 'test': hparams.test_trials}
        if hparams.actions_separately:
            return ArmbenchVideoActionWiseDataset(hparams.data_root, trials[dataset_type], hparams, vid_transform=vid_transform, training=dataset_type=='train')
        else:
            return ArmbenchVideoDataset(hparams.data_root, trials[dataset_type], hparams, vid_transform=vid_transform, training=dataset_type=='train')
    elif hparams.dataset == 'armbench_img_pair':
        crop_size = 650
        if dataset_type == 'train':
            vid_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(crop_size, pad_if_needed=True),
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
            )
        else:
            vid_transform = transforms.Compose(
                    [
                        transforms.CenterCrop(crop_size),
                        transforms.Resize(224),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
            )
        trials = {'train': hparams.training_trials, 'val': hparams.val_trials, 'test': hparams.test_trials}
        if hparams.actions_separately:
            return ArmbenchImgPairActionWiseDataset(hparams.data_root, trials[dataset_type], hparams, vid_transform=vid_transform, training=dataset_type=='train')

def accumulate_armbench_results(batch, output, hparams):
    if hparams.dataset == "armbench_video":
        vid, robot_activity, lbl, video_path = batch

        outputs = {}
        trial_names = [os.path.basename(vp)[:-4] for vp in video_path]
        robot_action = [rr[0] for rr in robot_activity.detach().cpu()]
        pred = torch.sigmoid(output)
        pred_bin = torch.where(pred < 0.5, 0, 1)
        per_vid_predictions = pred_bin.detach().cpu()
        outputs['cls_predictions'] = per_vid_predictions
        outputs['cls_gt'] = lbl.detach().cpu()
        outputs['logits'] = output.detach().cpu()
        outputs['trial_names'] = trial_names
        outputs['robot_actions'] = robot_action

        return outputs
    elif hparams.dataset == "armbench_img_pair":
        vid, robot_activity, lbl, video_path = batch
        outputs = {}
        pred = torch.sigmoid(output)
        pred_bin = torch.where(pred < 0.5, 0, 1)
        per_vid_predictions = pred_bin.detach().cpu()
        outputs['cls_predictions'] = per_vid_predictions
        outputs['cls_gt'] = lbl.detach().cpu()
        outputs['logits'] = output.detach().cpu()
        trial_names = [os.path.basename(vp)[:-4] for vp in video_path]
        robot_action = [rr for rr in robot_activity.detach().cpu()]
        outputs['robot_actions'] = robot_action
        outputs['trial_names'] = trial_names
        return outputs

def compute_armbench_metrics(outputs, result_type='val'):
    results = {}
    predictions = torch.cat([out['cls_predictions'] for out in outputs])
    gt = torch.cat([out['cls_gt'] for out in outputs])
    trial_names = [out['trial_names'] for out in outputs]
    trial_names = list(itertools.chain(*trial_names))
    robot_actions = [out['robot_actions'] for out in outputs]
    robot_actions = np.array(list(itertools.chain(*robot_actions)))
    logits = torch.cat([out['logits'] for out in outputs])

    if 'cls_scores' in outputs[0]:
        decon_scores = torch.cat([out['cls_scores'] for out in outputs])[gt[:, 0] == 1].mean(axis=1)[0]
        open_scores = torch.cat([out['cls_scores'] for out in outputs])[gt[:, 1] == 1].mean(axis=1)[1]
        results['%s_decons_score' % result_type] = decon_scores
        results['%s_open_score' % result_type] = open_scores

    conf_matrix = sklearn.metrics.multilabel_confusion_matrix(gt, predictions)

    tn = conf_matrix[:, 0, 0]
    tp = conf_matrix[:, 1, 1]
    fn = conf_matrix[:, 1, 0]
    fp = conf_matrix[:, 0, 1]

    recall = tp / (tp + fn)
    recall = np.nan_to_num(recall)

    fpr = fp / (fp + tn)

    results['%s_decons_recall' % result_type] = recall[label_encoder['deconstruction']-1]
    results['%s_decons_fpr' % result_type] = fpr[label_encoder['deconstruction']-1]

    results['%s_open_recall' % result_type] = recall[label_encoder['open']-1]
    results['%s_open_fpr' % result_type] = fpr[label_encoder['open']-1]

    f1_score = sklearn.metrics.f1_score(gt, predictions, average='weighted')
    results['%s_f1_score' % result_type] = f1_score

    return results, gt, predictions, trial_names, logits, robot_actions
