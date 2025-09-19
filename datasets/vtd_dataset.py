import os
import glob
import json
import numpy as np
import pdb
import math
from scipy import signal
import sklearn.metrics
import itertools
from torchvision.transforms import v2 as transforms
import decord
decord.bridge.set_bridge('torch')
import cv2

import torch

from datasets.utils import (
        variable_fps_frame_selection,
        constant_fps_frame_selection,
)

VIDEO_FPS = 18.0
SENSOR_FPS = 16.67
SENSOR_TIME = 24.0 # 400 samples at 16.67 Hz
NUM_UPSAMPLES = int(SENSOR_TIME * VIDEO_FPS) # 24 * 18 = 432
action_classes = {"pre-grasp": 0, "grasp": 1, "lift": 2, "release": 3}
action_classes_txt = {item:key for key,item in action_classes.items()}

def load_data(data_root, samples, lazy_loading=True, use_i3d=True, tactile_data_type='image', hparams=None, training=False):
    data_paths = {}
    data_actions = {}
    data_vid_feat = {}
    data_defect_mask = {}
    data_joint_pos = {}
    data_tactile = {}
    data_joint_pos_1d = {}
    data_tactile_1d = {}
    data_video_sensor_offset = {}
    data_video_paths = {}
    labels = {}
    data_trials = []
    # used for test time augmentation; trials are referred to multiple times
    data_trial_action = []
    for sample in samples:
        trial = sample
        sample_root = os.path.join(data_root, sample[1:])
        data_paths[trial] = sample_root

        data_video_paths[trial] = os.path.join(sample_root, 'front_rgb.mp4')

        label = np.loadtxt(os.path.join(sample_root, 'label.txt'))
        stage1 = int(label[0] * (VIDEO_FPS / SENSOR_FPS))
        stage2 = int(label[1] * (VIDEO_FPS / SENSOR_FPS))
        stage3 = int(label[2] * (VIDEO_FPS / SENSOR_FPS))
        labels[trial] = 1 - int(label[3]) # nominal = 0, anomalous = 1

        # timestamp when the object is grasped based on the video
        grasp_time_video = int(open(os.path.join(sample_root, 'video_grasp_timestamp.txt')).read())
        video_sensor_offset = max(0, grasp_time_video - stage1)
        data_video_sensor_offset[trial] = video_sensor_offset


        with open(os.path.join(sample_root, 'anomaly_timestamp.txt'), 'r') as fp:
            timestamps = fp.read().splitlines()
            timestamps = [int(t) for t in timestamps]
            if timestamps[0] == -1:
                anomalous = False
                anomaly_start_frame = -1
                anomaly_end_frame = -1
            else:
                anomalous = True
                anomaly_start_frame = int(timestamps[0] * VIDEO_FPS) # Frame rate = 18fps
                anomaly_end_frame = int((timestamps[1] + 1) * VIDEO_FPS)

        i3d_rgb_path = os.path.join(sample_root,  'rgb_i3d.npy')
        i3d_flow_path = os.path.join(sample_root, 'flow_i3d.npy')
        resnet_path = os.path.join(sample_root, 'resnet3d.npy')
        if not lazy_loading:
            if use_i3d:
                i3d_rgb = np.load(i3d_rgb_path)
                i3d_flow = np.load(i3d_flow_path)
                i3d_flow = np.concatenate((i3d_flow[0, :][np.newaxis, :], i3d_flow))
                vid_feat = np.hstack((i3d_rgb, i3d_flow)) # T x 2048
            else:
                vid_feat = np.load(resnet_path)
            vid_feat = vid_feat[video_sensor_offset:]
            data_vid_feat[trial] = vid_feat
        else:
            if use_i3d:
                data_vid_feat[trial] = [i3d_rgb_path, i3d_flow_path]
            else:
                data_vid_feat[trial] = resnet_path
            i3d_rgb = np.load(i3d_rgb_path)

        actions = torch.zeros(i3d_rgb.shape[0], dtype=torch.int32)
        actions[:stage1] = 0
        actions[stage1:stage2] = 1
        actions[stage2:stage3] = 2
        actions[stage3:] = 3
        if video_sensor_offset != 0:
            actions = actions[:-video_sensor_offset]

        data_actions[trial] = actions
        defect_mask = torch.zeros(i3d_rgb.shape[0], dtype=torch.int32)
        if anomalous:
            defect_mask[anomaly_start_frame:anomaly_end_frame] = 1
        defect_mask = defect_mask[video_sensor_offset:]
        data_defect_mask[trial] = defect_mask

        video_length = actions.shape[0]

        if tactile_data_type == 'image':
            positions = construct_position_tensor(np.loadtxt(os.path.join(sample_root, 'pos.txt')))
            positions = torch.from_numpy(positions)
            data_joint_pos[trial] = positions

            tactile_data = construct_tactile_tensor(np.loadtxt(os.path.join(sample_root, 'tactile.txt')))
            tactile_data = torch.from_numpy(tactile_data)
            data_tactile[trial] = tactile_data
        else:
            positions = upsample_position(np.loadtxt(os.path.join(sample_root, 'pos.txt')))
            data_joint_pos_1d[trial] = torch.from_numpy(positions)
            tactile_data = upsample_tactile(np.loadtxt(os.path.join(sample_root, 'tactile.txt')))
            data_tactile_1d[trial] = torch.from_numpy(tactile_data)

        if (hparams.action_aligned_fps_aug and not training):
            if not hparams.action_subset_frame_selection:
                unique_action_ids = [0, 1, 2, 3, 4]
                for act_id in unique_action_ids:
                    data_trial_action.append(act_id)
                    data_trials.append(trial)
            else:
                unique_action_ids = [1, 2, 3]
                for act_id in unique_action_ids:
                    data_trial_action.append(act_id)
                    data_trials.append(trial)
                # normal frame rate
                data_trial_action.append(0)
                data_trials.append(trial)
        else:
            if not hparams.action_subset_frame_selection:
                data_trials.append(trial)
                data_trial_action.append(4)
            else:
                data_trials.append(trial)
                data_trial_action.append(0)

    data = {}
    data['path'] = data_paths
    data['video_path'] = data_video_paths
    data['action'] = data_actions
    data['vid_feat'] = data_vid_feat
    data['label'] = labels
    data['defect_mask'] = data_defect_mask
    data['joint_pos_img'] = data_joint_pos
    data['tactile_img'] = data_tactile
    data['joint_pos_1d'] = data_joint_pos_1d
    data['tactile_1d'] = data_tactile_1d
    data['video_sensor_offset'] = data_video_sensor_offset
    data['trials'] = data_trials
    data['trial_action'] = data_trial_action
    return data

def construct_tactile_tensor(tactile_data):
    # based on code in https://github.com/priteshgohil/Multimodal-Machine-Learning/
    TAC_IMAGE_INDEX = np.array([[12,8,9,10,11],
                                [13,0,1,2,3],
                                [15,4,5,6,7],
                                [14,8,9,10,11]])
    # maximum magnitude along each of the 16 channels
    TACTILE_MAX_MAGNITUDE = np.array([23655, 20662, 14496,  6475, 41133, 64793, 59317, 33177, 19897,
                                      62084, 49874, 29170, 42944, 14976, 12311, 14331])
    tactile_data = tactile_data.astype(np.float32)
    tactile_data = signal.resample(tactile_data, NUM_UPSAMPLES)
    tactile_data[tactile_data < 0.1] = 0.0
    tactile_data /= TACTILE_MAX_MAGNITUDE
    #tactile_data = (2 * tactile_data) - 1 # range of -1 to 1
    tactile_tensor = tactile_data[:, TAC_IMAGE_INDEX.ravel()].reshape(-1, 4, 5, 1)
    return tactile_tensor

def upsample_tactile(tactile_data):
    # maximum magnitude along each of the 16 channels
    TACTILE_MAX_MAGNITUDE = np.array([23655, 20662, 14496,  6475, 41133, 64793, 59317, 33177, 19897,
                                      62084, 49874, 29170, 42944, 14976, 12311, 14331])
    tactile_data = tactile_data.astype(np.float32)
    tactile_data = signal.resample(tactile_data, NUM_UPSAMPLES)
    tactile_data[tactile_data < 0.1] = 0.0
    tactile_data /= TACTILE_MAX_MAGNITUDE
    return tactile_data

def construct_position_tensor(position_data):
    # based on code in https://github.com/priteshgohil/Multimodal-Machine-Learning/
    POS_IMAGE_INDEX = np.array([[0,6,7,0],
                                [5,1,3,5],
                                [4,0,2,4],
                                [0,6,7,0]])
    # maximum magnitude along each of the 16 channels
    POS_MAX_MAGNITUDE = np.array([ 20,  91,  25, 109,  54,  87,  12,  21])
    position_data = position_data.astype(np.float32)
    position_data = signal.resample(position_data, NUM_UPSAMPLES)
    position_data[position_data < 0.1] = 0.0
    position_data /= POS_MAX_MAGNITUDE
    #position_data = (2 * position_data) - 1 # range of -1 to 1
    position_tensor = position_data[:, POS_IMAGE_INDEX.ravel()].reshape(-1, 4, 4, 1)
    # set corners to zero
    position_tensor[:, 0, 0, 0] = 0.
    position_tensor[:, 3, 0, 0] = 0.
    position_tensor[:, 0, 3, 0] = 0.
    position_tensor[:, 3, 3, 0] = 0.
    return position_tensor

def upsample_position(position_data):
    # maximum magnitude along each of the 16 channels
    POS_MAX_MAGNITUDE = np.array([ 20,  91,  25, 109,  54,  87,  12,  21])
    position_data = position_data.astype(np.float32)
    position_data = signal.resample(position_data, NUM_UPSAMPLES)
    position_data[position_data < 0.1] = 0.0
    position_data /= POS_MAX_MAGNITUDE
    return position_data

def get_video(video_path, hparams, sensor_offset, samples=None, video_id=None, index=None, num_frames_to_sample=64, training=False):
    vr = decord.VideoReader(video_path)
    if hparams.action_subset_frame_selection and not hparams.action_aligned_fps_aug:
        frame_seq = np.where(samples['action'][video_id] != 0)[0]
        start_frame = np.random.randint(0, 5) if training else 0
        selected_frames = frame_seq[np.round(np.linspace(start_frame, len(frame_seq)-1, num_frames_to_sample)).astype(int)] + sensor_offset
    elif hparams.action_aligned_fps_aug:
        actions = samples['action'][video_id]
        if hparams.action_subset_frame_selection:
            low_fps_action_counts = {1: int(0.25 * num_frames_to_sample), 2: int(0.25 * num_frames_to_sample), 3: int(0.25 * num_frames_to_sample)}
            unique_action_ids = [1, 2, 3]
            if training:
                while True:
                    selected_action = np.random.randint(0, 4)
                    if selected_action == 0:
                        break
                    if len(np.where(actions == selected_action)[0]) > 10:
                        break
            else:
                selected_action = samples['trial_action'][index]
            if selected_action == 0:
                selected_frames = constant_fps_frame_selection(
                        actions,
                        low_fps_action_counts,
                        unique_action_ids,
                        num_frames_to_sample=num_frames_to_sample,
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
                        num_frames_to_sample=num_frames_to_sample,
                        training=training
                )
            selected_frames = selected_frames + sensor_offset
        else:
            low_fps_action_counts = {0: int(0.2 * num_frames_to_sample), 1: int(0.2 * num_frames_to_sample), 2: int(0.2 * num_frames_to_sample), 3: int(0.2 * num_frames_to_sample)}
            unique_action_ids = [0, 1, 2, 3]
            if training:
                while True:
                    selected_action = np.random.randint(0, 5)
                    if selected_action == 4:
                        break
                    if len(np.where(actions == selected_action)[0]) > 10:
                        break
            else:
                selected_action = samples['trial_action'][index]
            if selected_action == 4:
                selected_frames = constant_fps_frame_selection(
                        actions,
                        low_fps_action_counts,
                        unique_action_ids,
                        num_frames_to_sample=num_frames_to_sample,
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
                        num_frames_to_sample=num_frames_to_sample,
                        training=training
                )
            selected_frames = selected_frames + sensor_offset
    else:
        start_frame = np.random.randint(0, 5) if training else 0
        selected_frames = np.round(np.linspace(start_frame + sensor_offset, len(vr)-1, num_frames_to_sample)).astype(int)
    vid = vr.get_batch(selected_frames)
    vid = vid.permute(0, 3, 1, 2) # T x C x H x W
    return vid, selected_frames

class VTDVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, hparams, samples_list_file, tactile_data_type='image', transform=None, training=True):
        if not os.path.exists(data_root):
            msg = "Folder does not exist: {}\n".format(data_root)
            raise RuntimeError(msg)
        self.data_root = data_root
        samples_list = np.genfromtxt(os.path.join(data_root, samples_list_file), dtype=str)
        self.lazy_loading = True
        self.use_i3d = True
        self.data = load_data(data_root, samples_list, lazy_loading=self.lazy_loading, use_i3d=self.use_i3d, tactile_data_type=tactile_data_type, hparams=hparams, training=training)
        self.num_action_classes = len(action_classes)
        self.training = training
        self.tactile_data_type = tactile_data_type
        self.clip_size = 64
        self.hparams = hparams
        self.transform = transform


    def __getitem__(self, index):
        video_id = self.data['trials'][index]
        trial_action = self.data['trial_action'][index]
        if self.tactile_data_type == 'image':
            joint_pos = self.data['joint_pos_img'][video_id]
            tactile = self.data['tactile_img'][video_id]
        elif self.tactile_data_type == '1d':
            joint_pos = self.data['joint_pos_1d'][video_id]
            tactile = self.data['tactile_1d'][video_id]
        robot_activity = self.data['action'][video_id]
        defect_mask = self.data['defect_mask'][video_id]
        sensor_offset = self.data['video_sensor_offset'][video_id]

        video_file = self.data['video_path'][video_id]
        clip, frame_ids = get_video(video_file, self.hparams, sensor_offset, num_frames_to_sample=self.clip_size, training=self.training, video_id=video_id, index=index, samples=self.data)

        joint_pos = joint_pos[frame_ids-sensor_offset]
        tactile = tactile[frame_ids-sensor_offset]
        if self.hparams.vtd_tactile_data_type == '1d':
            joint_pos = joint_pos.permute(1, 0) # C x T
            tactile = tactile.permute(1, 0) # C x T
        elif self.hparams.vtd_tactile_data_type == 'image':
            joint_pos = joint_pos.permute(3, 0, 1, 2) # C x T x H x W
            tactile = tactile.permute(3, 0, 1, 2) # C x T x H x W

        if self.transform is not None:
            clip = clip.to(dtype=torch.get_default_dtype()).div(255)
            clip = self.transform(clip)
            clip = clip.permute((1, 0, 2, 3)).contiguous() # C, T, H, W

        labels = self.data['label'][video_id]
        return clip, joint_pos, tactile, labels, self.data['path'][video_id], trial_action

    def __len__(self):
        return len(self.data['trials'])

def get_vtd_dataset(hparams, dataset_type='train'):
    trials = {'train': hparams.training_trials, 'val': hparams.val_trials, 'test': hparams.test_trials}
    if dataset_type == 'train':
        vid_transform = transforms.Compose(
                [
                    transforms.RandomCrop(448),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize(224),
                ]
        )
    else:
        vid_transform = transforms.Compose(
                [
                    transforms.CenterCrop(448),
                    transforms.Resize(224),
                ]
        )
    return VTDVideoDataset(hparams.data_root, hparams, trials[dataset_type], tactile_data_type=hparams.vtd_tactile_data_type, transform=vid_transform, training=dataset_type=='train')

def accumulate_vtd_results(batch, output, hparams):
    clip, joint_pos, tactile, label, path, trial_action = batch
    per_vid_predictions = torch.argmax(output, axis=1).detach().cpu()
    outputs = {}
    outputs['cls_logits'] = output.detach().cpu()
    outputs['cls_predictions'] = per_vid_predictions
    outputs['cls_gt'] = label.detach().cpu()
    outputs['trial_names'] = path
    outputs['robot_actions'] = trial_action.detach().cpu()

    return outputs

def compute_vtd_metrics(outputs, result_type='val'):
    predictions = torch.cat([out['cls_predictions'] for out in outputs])
    gt = torch.cat([out['cls_gt'] for out in outputs])
    logits = torch.cat([out['cls_logits'] for out in outputs])

    trial_names = [out['trial_names'] for out in outputs]
    trial_names = list(itertools.chain(*trial_names))
    robot_actions = [out['robot_actions'] for out in outputs]
    robot_actions = np.array(list(itertools.chain(*robot_actions)))

    if 4 in robot_actions:
        nominal_fps_gt = gt[robot_actions == 4].numpy()
        nominal_fps_pred = predictions[robot_actions == 4].numpy()
        f1_score = sklearn.metrics.f1_score(nominal_fps_gt, nominal_fps_pred, average='weighted')
        recall_score = sklearn.metrics.recall_score(nominal_fps_gt, nominal_fps_pred, average='weighted')
        precision_score = sklearn.metrics.precision_score(nominal_fps_gt, nominal_fps_pred, average='weighted')
    else:
        nominal_fps_gt = gt[robot_actions == 0].numpy()
        nominal_fps_pred = predictions[robot_actions == 0].numpy()
        f1_score = sklearn.metrics.f1_score(nominal_fps_gt, nominal_fps_pred, average='weighted')
        recall_score = sklearn.metrics.recall_score(nominal_fps_gt, nominal_fps_pred, average='weighted')
        precision_score = sklearn.metrics.precision_score(nominal_fps_gt, nominal_fps_pred, average='weighted')

    logits = logits.numpy()

    results = {}
    results['%s_f1_score' % result_type] = f1_score
    results['%s_recall' % result_type] = recall_score
    results['%s_precision' % result_type] = precision_score
    results['%s_gt' % result_type] = gt
    results['%s_logits' % result_type] = logits
    results['%s_predictions' % result_type] = predictions
    results['%s_trial_names' % result_type] = trial_names

    if np.unique(robot_actions).shape[0] > 1:
        unique_actions = np.unique(robot_actions)
        aug_logits = None
        for act in unique_actions:
            if aug_logits is None:
                aug_logits = logits[robot_actions == act]
            else:
                aug_logits += logits[robot_actions == act]
        aug_logits /= len(unique_actions)
        aug_pred = np.argmax(aug_logits, axis=1)
        aug_f1_score = sklearn.metrics.f1_score(nominal_fps_gt, aug_pred, average='weighted')
        aug_recall_score = sklearn.metrics.recall_score(nominal_fps_gt, aug_pred, average='weighted')
        aug_precision_score = sklearn.metrics.precision_score(nominal_fps_gt, aug_pred, average='weighted')
        results['%s_aug_f1_score' % result_type] = aug_f1_score
        results['%s_aug_recall' % result_type] = aug_recall_score
        results['%s_aug_precision' % result_type] = aug_precision_score
        results['%s_aug_logits' % result_type] = aug_logits
    return results
