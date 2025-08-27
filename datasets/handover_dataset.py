import os
import glob
import json
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sklearn

import decord
decord.bridge.set_bridge('torch')

import cv2
import itertools

from torchvision.transforms import v2 as transforms
import torch

from datasets.utils import (
        variable_fps_frame_selection,
        constant_fps_frame_selection,
)

ALL = 'all'
r2h = 'robot to human handover'
h2r = 'human to robot handover'
KINOVA = 'Kinova Gen3'
HSR = 'Toyota HSR'


def get_gripper_state(robot_type, joint_states):
    '''
    -0.5: opened
    0.0: partially closed/open
    0.5: closed

    '''
    gripper_states = np.zeros(joint_states.shape[0])
    if robot_type == HSR:
        hand_motor = joint_states[7]
        gripper_states[joint_states[:, 7] > 0.95] = -0.5 # opened
        gripper_states[joint_states[:, 7] < -0.83] = 0.5 # closed
        # remaining = partially closed
    if robot_type == KINOVA:
        finger_joint = joint_states[7]
        gripper_states[joint_states[:, 7] < 0.1] = -0.5 # opened
        gripper_states[joint_states[:, 7] > 0.78] = 0.5 # closed
        # remaining = partially closed
    return gripper_states


def load_data(data_root, hparams, label_root=None, robot_type=ALL, task_type=ALL, training=True):
    trials = sorted(glob.glob(data_root + '/*'))
    data_robot_type = {}
    data_task_type = {}
    data_video_path = {}
    data_wrench_aligned = {}
    data_robot_actions = {}
    data_label = {}
    data_human_activity = {}
    data_gripper_state = {}
    data_flow = {}
    data_frame_sequences = {}
    data_trials = []
    # used for test time augmentation; trials are referred to multiple times
    data_trial_action = []

    for trial in trials:
        info_file = os.path.join(trial, 'task_info.json')
        with open(info_file, 'r') as fp:
            task_info = json.load(fp)
        if robot_type != ALL:
            if robot_type != task_info['robot']:
                continue
        if task_type != ALL:
            if task_type != task_info['task']:
                continue

        with open(os.path.join(label_root, os.path.basename(trial) + '.json')) as fp:
            label_data = json.load(fp)

        data_label[trial] = label_data['outcome']
        data_robot_type[trial] = task_info['robot']
        data_task_type[trial] = task_info['task']

        data_video_path[trial] = os.path.join(trial, 'head_cam.mp4')

        flow_files = np.array(sorted(glob.glob(os.path.join(trial, 'flow') + '/*_x.jpg')))
        data_flow[trial] = flow_files

        robot_actions = np.load(os.path.join(trial, 'robot_actions.npy'))
        if hparams.action_subset_frame_selection:
            selected_frames = np.where((robot_actions > 0) & (robot_actions < 4))[0]
            data_frame_sequences[trial] = selected_frames
        else:
            data_frame_sequences[trial] = []

        aligned_wrench = np.load(os.path.join(trial, 'wrench_resampled.npy'))
        aligned_joint_pos = np.load(os.path.join(trial, 'joint_pos_resampled.npy'))

        with open(os.path.join(os.path.dirname(data_root), 'wrench_stats.json'), 'r') as fp:
            wrench_stats = json.load(fp)
        mean = np.array(wrench_stats[task_info['robot']]['mean'])
        std = np.array(wrench_stats[task_info['robot']]['std'])
        aligned_wrench = np.divide((aligned_wrench - mean), std)


        data_wrench_aligned[trial] = aligned_wrench
        gripper_state = get_gripper_state(task_info['robot'], aligned_joint_pos)
        data_gripper_state[trial] = gripper_state
        data_robot_actions[trial] = robot_actions
        human_activity_state = np.load(os.path.join(trial, 'human_activity.npy'))
        data_human_activity[trial] = human_activity_state
        if hparams.action_aligned_fps_aug and not training:
            unique_action_ids = [1, 2, 3]
            for act_id in unique_action_ids:
                data_trial_action.append(act_id)
                data_trials.append(trial)
            # normal frame rate
            data_trial_action.append(0)
            data_trials.append(trial)
        else:
            data_trials.append(trial)
            data_trial_action.append(0)
    data = {}
    data['robot'] = data_robot_type
    data['task'] = data_task_type
    data['video'] = data_video_path
    data['frame_seq'] = data_frame_sequences
    data['flow'] = data_flow
    data['wrench_aligned'] = data_wrench_aligned
    data['label'] = data_label
    data['robot_actions'] = data_robot_actions
    data['gripper_state'] = data_gripper_state
    data['human_activity'] = data_human_activity
    data['trials'] = data_trials
    data['trial_action'] = data_trial_action
    return data

def get_video(video_path, hparams, frame_seq, training=True, samples=None, video_id=None, index=None, load_video=True, num_frames_to_sample=64):
    vr = decord.VideoReader(video_path)
    if len(frame_seq):
        # we just have one array of frame ids
        if isinstance(frame_seq, np.ndarray):
            start_frame = np.random.randint(0, 5) if training else 0
            selected_frames = frame_seq[np.round(np.linspace(start_frame, len(frame_seq)-1, num_frames_to_sample)).astype(int)]
    elif hparams.action_aligned_fps_aug:
        actions = samples['robot_actions'][video_id]
        low_fps_action_counts = {1: int(0.25 * num_frames_to_sample), 2: int(0.375 * num_frames_to_sample), 3: int(0.25 * num_frames_to_sample)}
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
    else:
        start_frame = np.random.randint(0, 5) if training else 0
        selected_frames = np.round(np.linspace(start_frame, len(vr)-1, num_frames_to_sample)).astype(int)
    if not load_video:
        return selected_frames
    vid = vr.get_batch(selected_frames)
    blank_images_idx = np.where(selected_frames == -1)[0]
    if len(blank_images_idx) > 0:
        vid[blank_images_idx] = 0
    vid = vid.permute(0, 3, 1, 2) # T x C x H x W
    return vid, selected_frames

def flow_loader_frame_ids(flow_files, frame_ids):
    path_root = os.path.dirname(flow_files[0])
    all_frames = []
    for ff in frame_ids:
        if ff > len(flow_files):
            print("Trying to load optical flow file that doesn't exist. %d %s" % (ff, flow_files[-1]))
            exit(0)
        if ff == 0:
            flow_x = np.ones((480, 640), dtype=np.uint8) * 255
            flow_y = np.ones((480, 640), dtype=np.uint8) * 255
        else:
            flow_x = os.path.join(path_root, 'frame%04d_x.jpg' % ff)
            flow_y = os.path.join(path_root, 'frame%04d_y.jpg' % ff)
            flow_x = cv2.imread(flow_x, cv2.IMREAD_GRAYSCALE)
            flow_y = cv2.imread(flow_y, cv2.IMREAD_GRAYSCALE)
        flow = np.stack((flow_x, flow_y), axis=2)
        all_frames.append(flow)
    all_frames = np.array(all_frames)
    all_frames = torch.from_numpy(all_frames).permute(0, 3, 1, 2) # T x C x H x W
    return all_frames


class HandoverDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, hparams, label_root=None, robot_type=ALL, task_type=ALL, transform = None, training=False, num_classes=4):
        self.data_root = data_root
        self.data = load_data(data_root, hparams, label_root, robot_type, task_type, training=training)
        if len(self.data['video']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)

        self.hparams = hparams
        self.robot_type = robot_type
        self.task_type = task_type
        self.training = training
        self.transform = transform
        self.clip_size = 64
        self.num_classes = num_classes

    def __getitem__(self, index):
        video_id = self.data['trials'][index]
        video_file = self.data['video'][video_id]

        wrench = self.data['wrench_aligned'][video_id]
        wrench = torch.from_numpy(wrench).float()

        gripper_state = self.data['gripper_state'][video_id]
        gripper_state = torch.from_numpy(gripper_state).unsqueeze(1).float()

        robot_actions = self.data['robot_actions'][video_id]
        frame_seq = self.data['frame_seq'][video_id]
        if self.hparams.video_type == 'rgb':
            clip, frame_ids = get_video(video_file, self.hparams, frame_seq, training=self.training, samples=self.data, video_id=video_id, index=index, num_frames_to_sample=self.clip_size)
        elif self.hparams.video_type == 'flow':
            frame_ids = get_video(video_file, self.hparams, frame_seq, training=self.training, samples=self.data, video_id=video_id, index=index, load_video=False, num_frames_to_sample=self.clip_size)
            clip = flow_loader_frame_ids(self.data['flow'][video_id], frame_ids)
        wrench = wrench[frame_ids]
        gripper_state = gripper_state[frame_ids]

        # T x C
        wrench = wrench.permute(1, 0)
        gripper_state = gripper_state.permute(1, 0)

        if self.transform is not None:
            clip = clip.to(dtype=torch.get_default_dtype()).div(255)
            clip = self.transform(clip)
            clip = clip.permute((1, 0, 2, 3)).contiguous() # C, T, H, W
        label = self.data['label'][video_id]

        trial_action = self.data['trial_action'][index]

        return clip, wrench, gripper_state, label, video_id, trial_action

    def __len__(self):
        return len(self.data['trials'])

def get_handover_dataset(hparams, dataset_type='train'):
    dataset_root = {'train': os.path.join(hparams.data_root, 'training_set'), 'val': os.path.join(hparams.data_root, 'val_set'), 'test': os.path.join(hparams.data_root, 'test_set')}
    label_root = {'train': os.path.join(hparams.data_root, 'training_labels'), 'val': os.path.join(hparams.data_root, 'val_labels'), 'test': os.path.join(hparams.data_root, 'test_labels')}

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
    if hparams.dataset == 'handover_video':
        return HandoverDataset(dataset_root[dataset_type], hparams, label_root=label_root[dataset_type], training=dataset_type=='train', transform=vid_transform)


def accumulate_handover_results(batch, output, hparams):
    outputs = {}
    vid, wrench, gripper_state, label, video_id, trial_action = batch
    per_vid_predictions = torch.argmax(output, axis=1).detach().cpu()
    outputs['cls_logits'] = output.detach().cpu()
    outputs['cls_predictions'] = per_vid_predictions
    outputs['cls_gt'] = label.detach().cpu()
    outputs['trial_names'] = video_id
    outputs['robot_actions'] = trial_action.detach().cpu()

    return outputs

def compute_handover_metrics(outputs, result_type='val'):
    predictions = torch.cat([out['cls_predictions'] for out in outputs])
    gt = torch.cat([out['cls_gt'] for out in outputs])
    logits = torch.cat([out['cls_logits'] for out in outputs])

    trial_names = [out['trial_names'] for out in outputs]
    trial_names = list(itertools.chain(*trial_names))
    robot_actions = [out['robot_actions'] for out in outputs]
    robot_actions = np.array(list(itertools.chain(*robot_actions)))

    nominal_fps_gt = gt[robot_actions == 0]
    nominal_fps_pred = predictions[robot_actions == 0]
    accuracy = sklearn.metrics.accuracy_score(nominal_fps_gt, nominal_fps_pred)

    results = {}
    results['%s_accuracy' % result_type] = accuracy
    results['%s_logits' % result_type] = logits
    results['%s_gt' % result_type] = nominal_fps_gt
    results['%s_predictions' % result_type] = predictions
    results['%s_trial_names' % result_type] = trial_names
    results['%s_robot_actions' % result_type] = robot_actions

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
        aug_accuracy = sklearn.metrics.accuracy_score(nominal_fps_gt.numpy(), aug_pred.numpy())
        results['%s_aug_accuracy' % result_type] = aug_accuracy
        results['%s_aug_logits' % result_type] = aug_logits.numpy()
    return results
