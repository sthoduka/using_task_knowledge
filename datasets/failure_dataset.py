import os
import glob
import itertools
import numpy as np
import pdb
import sklearn.metrics

from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
import torch
from torchvision.io import read_image

from datasets.utils import (
        variable_fps_frame_selection,
        constant_fps_frame_selection,
)
sub_action_classes = {"approach": 0, "act": 1, "retract": 2}

action_classes = {"place": 0, "pour": 1, "push": 2, "put_in": 3, "put_on": 4}
actions = ['place', 'pour', 'push', 'put_in', 'put_on']

def make_failure_dataset_video_only(directory, hparams):
    data = {}
    actions = ['place', 'pour', 'push', 'put_in', 'put_on']
    annotations = {}
    with open(os.path.join(directory, 'training_data.txt'), 'r') as fp:
        train_paths = fp.readlines()
    train_paths = [tp.strip() for tp in train_paths]

    with open(os.path.join(directory, 'test_data.txt'), 'r') as fp:
        test_paths = fp.readlines()
    test_paths = [tp.strip() for tp in test_paths]

    for action in actions:
        annotations[action] = np.genfromtxt(os.path.join(directory, 'annotation', '%s_annotation.txt' % action), delimiter=',', skip_header=1)
    ds_actions = []
    ds_trials = []
    ds_annotations = []
    ds_boxes = []
    ds_box_frame_sequences = []
    ds_all_boxes = []
    ds_img_paths = []
    ds_audio_path = []
    ds_video_ids = []
    ds_robot_activity = []
    ds_data_frame_sequences = []
    ds_train_ids = []
    ds_test_ids = []
    vid_id = 0

    for action in actions:
        trials = sorted(glob.glob(os.path.join(directory, 'failnet_dataset', 'rgb_imgs', action) + '/*'))
        for trial in trials:
            trial_id = int(os.path.basename(trial))
            # 1 = failure, 0 = success
            ann = int(annotations[action][annotations[action][:, 0] == trial_id][0][1])
            train = False
            if trial in train_paths:
                train = True
                ds_train_ids.append(vid_id)
            elif trial in test_paths:
                ds_test_ids.append(vid_id)

            robot_activity_path = os.path.join(directory, 'failnet_dataset', 'action_segmentation', action, str(trial_id) + '.npy')
            robot_activity = np.load(robot_activity_path)
            ds_data_frame_sequences.append([])

            ds_img_paths.append(trial)
            ds_actions.append(action)
            ds_annotations.append(ann)
            ds_trials.append(trial_id)
            ds_video_ids.append(vid_id)
            ds_robot_activity.append(robot_activity)
            vid_id += 1

    data['action'] = np.array(ds_actions)
    data['annotation'] = np.array(ds_annotations)
    data['trial'] = np.array(ds_trials)
    data['img_path'] = np.array(ds_img_paths)
    data['video_id'] = np.array(ds_video_ids)
    data['robot_activity'] = ds_robot_activity
    data['frame_seq'] = ds_data_frame_sequences

    data['train_ids'] = ds_train_ids
    data['test_ids'] = ds_test_ids

    return data

def load_image_pair(path, frame_seq, training=False):
    imgs = sorted(glob.glob(path + '/*.png'))
    torch_imgs = []

    for frame_id in frame_seq:
        im = read_image(imgs[frame_id])
        torch_imgs.append(im)
    torch_imgs = torch.stack(torch_imgs)
    return torch_imgs

def load_images(path, hparams, frame_seq, training=False, samples=None, video_id=None):
    imgs = sorted(glob.glob(path + '/*.png'))
    num_frames_to_sample = 32
    if len(frame_seq):
        # we just have one array of frame ids
        if isinstance(frame_seq, np.ndarray):
            if training:
                start_frame = np.random.randint(0, 5)
            else:
                start_frame = 0
            selected_frames = frame_seq[np.round(np.linspace(start_frame, len(frame_seq)-1, num_frames_to_sample)).astype(int)]
    elif hparams.action_aligned_fps_aug and (training or hparams.selected_action != ''):
        actions = samples['robot_activity'][video_id]
        low_fps_action_counts = {0: 8, 1: 8, 2: 8}
        unique_action_ids = [0, 1, 2]
        if hparams.selected_action != '':
            high_fps_action = int(hparams.selected_action)
        else:
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
    else:
        if training:
            start_frame = np.random.randint(0, 5)
        else:
            start_frame = 0
        selected_frames = np.round(np.linspace(start_frame, len(imgs)-1, num_frames_to_sample)).astype(int)
    torch_imgs = []

    for idx, img in enumerate(imgs):
        im = read_image(img)
        torch_imgs.append(im)
    torch_imgs = torch.stack(torch_imgs)
    torch_imgs = torch_imgs[selected_frames]
    blank_images_idx = np.where(selected_frames == -1)[0]
    if len(blank_images_idx) > 0:
        torch_imgs[blank_images_idx] = 0
    return torch_imgs, selected_frames


class FailureVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, hparams, dataset_type='train', vid_transform=None):
        self.root = root
        samples = make_failure_dataset_video_only(self.root, hparams)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.vid_transform = vid_transform
        self.samples = samples
        self.dataset_type = dataset_type
        self.actions = ['place', 'pour', 'push', 'put_in', 'put_on']

        self.hparams = hparams

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            video_id = self.samples['train_ids'][index]
        elif self.dataset_type == 'val':
            video_id = self.samples['test_ids'][index]
        elif self.dataset_type == 'test':
            video_id = self.samples['test_ids'][index]

        path = self.samples['img_path'][video_id]

        frame_seq = self.samples['frame_seq'][video_id]

        vid, selected_frames = load_images(path, self.hparams, frame_seq, training=self.dataset_type=='train', samples=self.samples, video_id=video_id)
        vid = self.vid_transform(vid)
        vid = vid.permute(1, 0, 2, 3).contiguous() # C x T x H x W

        action_index = self.actions.index(self.samples['action'][video_id])
        label = self.samples['annotation'][video_id]

        robot_activity = torch.from_numpy(self.samples['robot_activity'][video_id])[selected_frames]
        return vid, robot_activity, action_index, label, path

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.samples['train_ids'])
        elif self.dataset_type == 'val':
            return len(self.samples['test_ids'])
        elif self.dataset_type == 'test':
            return len(self.samples['test_ids'])

class FailureImgPairDataset(torch.utils.data.Dataset):
    def __init__(self, root, hparams, dataset_type='train', vid_transform=None):
        self.root = root
        samples = make_failure_dataset_video_only(self.root, hparams)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.vid_transform = vid_transform
        self.samples = samples
        self.dataset_type = dataset_type
        self.actions = ['place', 'pour', 'push', 'put_in', 'put_on']

        self.hparams = hparams

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            video_id = self.samples['train_ids'][index]
        elif self.dataset_type == 'val':
            video_id = self.samples['test_ids'][index]
        elif self.dataset_type == 'test':
            video_id = self.samples['test_ids'][index]

        path = self.samples['img_path'][video_id]

        robot_activity = self.samples['robot_activity'][video_id]
        approach_frames = np.where(robot_activity == 0)[0]
        act_frames = np.where(robot_activity == 1)[0]
        retract_frames = np.where(robot_activity == 2)[0]
        if len(approach_frames) == 0:
            approach_frames = act_frames
        if len(retract_frames) == 0:
            retract_frames = act_frames

        num_frames = len(robot_activity)
        if self.dataset_type == 'train':
            first_frame = np.random.choice(approach_frames[:len(approach_frames) //2])
            last_frame = np.random.choice(retract_frames[len(retract_frames) // 2:])
            frames = [first_frame, last_frame]
        else:
            frames = [approach_frames[0], retract_frames[-1]]


        vid = load_image_pair(path, frames, training=self.dataset_type=='train')
        vid = self.vid_transform(vid) # T x C x H x W

        action_index = self.actions.index(self.samples['action'][video_id])
        label = self.samples['annotation'][video_id]
        return vid, action_index, label, path

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.samples['train_ids'])
        elif self.dataset_type == 'val':
            return len(self.samples['test_ids'])
        elif self.dataset_type == 'test':
            return len(self.samples['test_ids'])

def get_failure_dataset(hparams, dataset_type='train'):
    crop_size = 480
    if hparams.dataset == 'failure_video':
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
        return FailureVideoDataset(hparams.data_root, hparams, dataset_type, vid_transform=vid_transform)
    elif hparams.dataset == 'failure_img_pair':
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
        return FailureImgPairDataset(hparams.data_root, hparams, dataset_type, vid_transform=vid_transform)

def accumulate_failure_results(batch, output, hparams):
    feats, imgs, mfcc, def_seg, robot_activity, output_mask, label, action_index, path = batch
    cls_out = output

    _, cls_pred = torch.max(cls_out[-1], 1)
    per_vid_cls_pred = []
    per_vid_cls_gt = []
    per_vid_action = []
    per_vid_trial = []
    for vididx, cpred in enumerate(cls_pred):
        per_vid_cls_pred.append(cpred[output_mask[vididx, 0, :] == 1][-1].cpu().item())
        per_vid_cls_gt.append(label[vididx].cpu().item())
        per_vid_action.append(actions[action_index[vididx]])
        per_vid_trial.append(os.path.basename(path[vididx]))
    outputs = {}
    outputs['cls_pred'] = per_vid_cls_pred
    outputs['cls_gt'] = per_vid_cls_gt
    outputs['actions'] = per_vid_action
    outputs['trials'] = per_vid_trial
    return outputs

def accumulate_failure_video_results(batch, output, hparams):
    outputs = {}
    if hparams.dataset == 'failure_video':
        vid, robot_activity, action_idx, label, path = batch
    elif hparams.dataset == 'failure_img_pair':
        vid, action_idx, label, path = batch

    pred = torch.argmax(output, 1).detach().cpu().numpy()
    gt = label.detach().cpu().numpy()
    outputs['logits'] = output.detach().cpu()
    trial_names = [os.path.join(os.path.basename(os.path.dirname(pa)), os.path.basename(pa)) for pa in path]
    outputs['cls_pred'] = pred
    outputs['cls_gt'] = gt
    outputs['trial_names'] = trial_names
    return outputs


def compute_failure_metrics(outputs, result_type='val'):
    gt = []
    predictions = []
    logits = torch.cat([out['logits'] for out in outputs])
    trial_names = [out['trial_names'] for out in outputs]
    trial_names = list(itertools.chain(*trial_names))
    for batch in outputs:
        gt.extend(batch['cls_gt'])
        predictions.extend(batch['cls_pred'])

    f1_score = sklearn.metrics.f1_score(gt, predictions, average='weighted')
    recall_score = sklearn.metrics.recall_score(gt, predictions, average='weighted')
    precision_score = sklearn.metrics.precision_score(gt, predictions, average='weighted')

    results = {}
    results['%s_f1_score' % result_type] = f1_score
    results['%s_recall' % result_type] = recall_score
    results['%s_precision' % result_type] = precision_score

    return results, np.array(gt), np.array(predictions), logits.numpy(), trial_names
