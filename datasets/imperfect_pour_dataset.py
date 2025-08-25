import json
import os
import pdb
import itertools
import numpy as np
import sklearn.metrics
from torchvision.transforms import v2 as transforms
import torch
import decord
decord.bridge.set_bridge('torch')

from datasets.utils import (
        variable_fps_frame_selection,
        constant_fps_frame_selection,
)

action_classes = {"pre": 0, "core": 1, "post": 2}

task_classes = {'pick': 0, 'pour': 1, 'place': 2, 'wipe': 3}

def load_data(data_root, trials, hparams, training):
    data_video_path = []
    data_video_id = []
    labels = []
    data_task_label = []
    data_task_id = []
    data_actions = []
    data_frame_range = []
    data_has_spill = []
    data_missing_object = []
    data_has_fallen = []

    data = {}
    for trial in trials:
        json_file = os.path.join(data_root, 'annotations', trial)
        with open(json_file, 'r') as fp:
            sub_trials = json.load(fp)
        for sub_trial in sub_trials:
            data_task_label.append(sub_trial["action"])
            data_video_id.append(os.path.basename(trial)[:-5])
            data_video_path.append(os.path.join(data_root, 'videos', data_video_id[-1], data_video_id[-1] + '.mp4'))
            data_frame_range.append(np.array(range(sub_trial["pre_start_fid"], sub_trial["post_end_fid"])))
            actions = np.zeros_like(data_frame_range[-1]) # pre
            start_fid = sub_trial["pre_start_fid"]
            if sub_trial["pre_start_fid"] == sub_trial["post_start_fid"] and sub_trial["pre_end_fid"] == sub_trial["post_end_fid"]:
                pass
            else:
                actions[sub_trial["pre_end_fid"]-start_fid:sub_trial["post_start_fid"]-start_fid] = 1 # core
                actions[sub_trial["post_start_fid"]-start_fid:] = 2 # post
            data_actions.append(actions)
            if 'has_spill' in sub_trial:
                data_has_spill.append(sub_trial['has_spill'])
            else:
                data_has_spill.append(False)
            if 'missing_object' in sub_trial:
                data_missing_object.append(sub_trial['missing_object'])
            else:
                data_missing_object.append(False)
            if 'has_fallen' in sub_trial:
                data_has_fallen.append(sub_trial['has_fallen'])
            else:
                data_has_fallen.append(False)
            if data_has_spill[-1] or data_missing_object[-1] or data_has_fallen[-1]:
                labels.append(1) # failure
            else:
                labels.append(0) # success
            if 'pick' in sub_trial["action"]:
                data_task_id.append(task_classes['pick'])
            if 'pour' in sub_trial["action"]:
                data_task_id.append(task_classes['pour'])
            if 'place' in sub_trial["action"]:
                data_task_id.append(task_classes['place'])
            if 'wipe' in sub_trial["action"]:
                data_task_id.append(task_classes['wipe'])
    data['task_label'] = np.array(data_task_label)
    data['video_path'] = np.array(data_video_path)
    data['video_id'] = np.array(data_video_id)
    data['frame_range'] = data_frame_range
    data['robot_activity'] = data_actions
    data['label'] = np.array(labels)
    data['task_id'] = np.array(data_task_id)
    data['has_spill'] = np.array(data_has_spill)
    data['has_fallen'] = np.array(data_has_fallen)
    data['missing_object'] = np.array(data_missing_object)
    return data

def get_video(video_path, frame_seq, hparams, training=True, data=None, index=None):
    vr = decord.VideoReader(video_path)
    num_frames_to_sample = 32
    if hparams.action_aligned_fps_aug and training:
        actions = data['robot_activity'][index]
        low_fps_action_counts = {0: 8, 1: 8, 2: 8}
        unique_action_ids = [0, 1, 2]
        if hparams.selected_action != '':
            high_fps_action = int(hparams.selected_action)
        while True:
            high_fps_action = np.random.randint(0, 4)
            if high_fps_action == 3: # constant fps
                break
            if len(np.where(actions == high_fps_action)[0]) > 5:
                break
        if high_fps_action == 3:
            selected_frames = constant_fps_frame_selection(
                    actions,
                    low_fps_action_counts,
                    unique_action_ids,
                    training=training
            )
        else:
            del low_fps_action_counts[high_fps_action]
            selected_frames = variable_fps_frame_selection(
                    actions,
                    high_fps_action,
                    low_fps_action_counts,
                    unique_action_ids,
                    training=training
            )
        selected_frames = frame_seq[selected_frames]
    else:
        if training:
            start_frame = np.random.randint(0, 5)
        else:
            start_frame = 0
        selected_frames = frame_seq[np.round(np.linspace(start_frame, len(frame_seq)-1, num_frames_to_sample)).astype(int)]
    vid = vr.get_batch(selected_frames)
    vid = vid.permute(0, 3, 1, 2) # T x C X H X W
    selected_frames = selected_frames - frame_seq[0]
    return vid, selected_frames

def get_img_pair(video_path, frame_seq, hparams, training=True, data=None, index=None):
    vr = decord.VideoReader(video_path)
    vid = vr.get_batch(frame_seq)
    vid = vid.permute(0, 3, 1, 2) # T x C X H X W
    return vid

class ImperfectPourDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_file, hparams, vid_transform=None, training=True):
        self.data_root = data_root
        with open(os.path.join(data_root, split_file), 'r') as fp:
            self.trials = fp.readlines()
        self.trials = [tp.strip() for tp in self.trials]
        self.data = load_data(data_root, self.trials, hparams, training)
        if len(self.data['label']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)
        self.training = training
        self.vid_transform = vid_transform
        self.hparams = hparams

    def __getitem__(self, index):
        label = self.data['label'][index]
        video_path = self.data['video_path'][index]
        frame_seq = self.data['frame_range'][index]
        task_id = self.data['task_id'][index]

        vid, selected_frames = get_video(video_path, frame_seq, self.hparams, training=self.training, data=self.data, index=index)

        if self.vid_transform is not None:
            vid = self.vid_transform(vid)
        vid = vid.permute(1, 0, 2, 3) # C X T x H x W
        robot_actions = self.data['robot_activity'][index][selected_frames]

        return vid, robot_actions, task_id, label, video_path

    def __len__(self):
        return len(self.data['label'])

class ImperfectPourImgPairDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_file, hparams, vid_transform=None, training=True):
        self.data_root = data_root
        with open(os.path.join(data_root, split_file), 'r') as fp:
            self.trials = fp.readlines()
        self.trials = [tp.strip() for tp in self.trials]
        self.data = load_data(data_root, self.trials, hparams, training)
        if len(self.data['label']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)
        self.training = training
        self.vid_transform = vid_transform
        self.hparams = hparams

    def __getitem__(self, index):
        label = self.data['label'][index]
        video_path = self.data['video_path'][index]
        frame_seq = self.data['frame_range'][index]
        task_id = self.data['task_id'][index]
        robot_actions = self.data['robot_activity'][index]
        pre = np.where(robot_actions == 0)[0]
        post = np.where(robot_actions == 2)[0]
        if len(post) == 0: # happens in case of missing objects
            post = pre

        if self.training:
            first_frame = np.random.choice(pre[:len(pre) // 2])
            last_frame = np.random.choice(post[len(post) // 2:])
            frames = [frame_seq[first_frame], frame_seq[last_frame]]
        else:
            frames = [frame_seq[pre[0]], frame_seq[post[-1]]]

        vid = get_img_pair(video_path, frames, self.hparams, training=self.training, data=self.data, index=index)

        if self.vid_transform is not None:
            vid = self.vid_transform(vid)


        return vid, task_id, label, video_path

    def __len__(self):
        return len(self.data['label'])

def get_imp_dataset(hparams, dataset_type='train'):
    crop_size = 640
    if hparams.dataset == 'imperfect_pour':
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
            split_file = 'train_split.txt'
        else:
            vid_transform = transforms.Compose(
                    [
                        transforms.CenterCrop(crop_size),
                        transforms.Resize(224),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
            )
            split_file = 'val_split.txt'
        return ImperfectPourDataset(hparams.data_root, split_file, hparams, vid_transform=vid_transform)
    elif hparams.dataset == 'imperfect_pour_img_pair':
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
            split_file = 'train_split.txt'
        else:
            vid_transform = transforms.Compose(
                    [
                        transforms.CenterCrop(crop_size),
                        transforms.Resize(224),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
            )
            split_file = 'val_split.txt'
        return ImperfectPourImgPairDataset(hparams.data_root, split_file, hparams, vid_transform=vid_transform)


def accumulate_imp_results(batch, output, hparams):
    if hparams.dataset == 'imperfect_pour':
        vid, robot_activity, action_idx, label, video_path = batch
    elif hparams.dataset == 'imperfect_pour_img_pair':
        vid, action_idx, label, video_path = batch
    outputs = {}
    pred = torch.argmax(output, 1).detach().cpu().numpy()
    gt = label.detach().cpu().numpy()
    trial_names = [os.path.basename(vp)[:-4] for vp in video_path]
    outputs['cls_pred'] = pred
    outputs['cls_gt'] = gt
    outputs['logits'] = output.detach().cpu()
    outputs['trial_names'] = trial_names
    return outputs

def compute_imp_metrics(outputs, result_type='val'):
    gt = []
    predictions = []
    for batch in outputs:
        gt.extend(batch['cls_gt'])
        predictions.extend(batch['cls_pred'])
        trial_names = [out['trial_names'] for out in outputs]
        trial_names = list(itertools.chain(*trial_names))
        logits = torch.cat([out['logits'] for out in outputs])
    f1_score = sklearn.metrics.f1_score(gt, predictions, average='weighted')
    recall_score = sklearn.metrics.recall_score(gt, predictions, average='weighted')
    precision_score = sklearn.metrics.precision_score(gt, predictions, average='weighted')
    results = {}
    results['%s_f1_score' % result_type] = f1_score
    results['%s_recall' % result_type] = recall_score
    results['%s_precision' % result_type] = precision_score

    return results, torch.tensor(gt), torch.tensor(predictions), trial_names, logits
