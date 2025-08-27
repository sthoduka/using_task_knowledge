import torch.nn as nn
import torch

from torchvision import models
import torch.nn.functional as F

from slowfast.models import build_model
from slowfast.utils.parser import load_config
import argparse

from models.pytorch_i3d import InceptionI3d, Unit3D

import pdb

class VideoClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        output_dim = 768
        mvit_args = argparse.Namespace()
        mvit_args.opts = None
        cfg = load_config(mvit_args, args.mvit_config)
        self.video_model = build_model(cfg, 0)
        self.video_model.load_state_dict(torch.load(args.mvit_ckpt)['model_state'])
        self.video_model.head = nn.Identity()
        self.classifier = nn.Linear(output_dim, args.num_classes, bias=True)

    def forward(self, batch):
        if 'failure' in self.hparams.dataset or 'imperfect_pour' in self.hparams.dataset:
            vid, robot_actions, action_lbl, lbl, _ = batch
        else:
            vid, robot_actions, lbl, _ = batch
        vid = (vid,) # mvit expects an array of inputs and takes the first element
        out = self.video_model(vid)
        out = self.classifier(out)
        return out

    def loss_function(self, output, batch):
        if 'failure' in self.hparams.dataset or 'imperfect_pour' in self.hparams.dataset:
            vid, robot_actions, action_lbl, lbl, _ = batch
        else:
            vid, robot_actions, lbl, _ = batch
        out = output

        if 'armbench' in self.hparams.dataset:
            loss = F.binary_cross_entropy_with_logits(out, lbl)
        elif 'failure' in self.hparams.dataset:
            loss = F.cross_entropy(out, lbl)
        elif 'imperfect_pour' in self.hparams.dataset:
            loss = F.cross_entropy(out, lbl)
        return loss

class ImgPairClassifier(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        output_dim = 512 * 2
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cls_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(output_dim, hparams.num_classes, bias=True)

    def forward(self, batch):
        if 'failure' in self.hparams.dataset or 'imperfect_pour' in self.hparams.dataset:
            img_pair, action_lbl, lbl, _ = batch
        else:
            img_pair, robot_action, lbl, trial_id = batch
        B, T, C, H, W = img_pair.size()
        img_pair = img_pair.view(B*T, C, H, W)
        out = self.cls_model(img_pair)
        out = out.squeeze(2).squeeze(2)
        out = out.view(B, -1)
        out = self.classifier(out)
        return out

    def loss_function(self, output, batch):
        if 'failure' in self.hparams.dataset or 'imperfect_pour' in self.hparams.dataset:
            img_pair, action_lbl, lbl, trial_id = batch
            out = output
            loss = F.cross_entropy(out, lbl)
        else:
            img_pair, robot_action, lbl, trial_id = batch
            loss = F.binary_cross_entropy_with_logits(output, lbl)
        return loss


class MultimodalClassifier(nn.Module):
    '''
    Original source: https://github.com/sthoduka/handover_failure_detection/blob/master/pytorch-i3d-trainer/i3d_trainer.py#L50
    '''

    def __init__(self, hparams, load_pretrained=True):
        super(MultimodalClassifier, self).__init__()
        if hparams.video_type == 'rgb':
            self.i3d = InceptionI3d(400, in_channels=3)
        else:
            self.i3d = InceptionI3d(400, in_channels=2)
        if load_pretrained:
            if hparams.i3d_model_path != '':
                print('Loading %s weights ' % hparams.i3d_model_path)
                self.i3d.load_state_dict(torch.load(hparams.i3d_model_path))
        self.i3d.replace_logits(hparams.num_classes, in_channels=1024+16+16)

        self.wrench_conv = nn.Sequential(
                                nn.Conv1d(6, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )
        self.gripper_conv = nn.Sequential(
                                nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )

    def forward(self, batch):
        inputs, wrench, gripper_state, labels, vidx, trial_action = batch
        i3d_feat = self.i3d.extract_features(inputs)
        wout = self.wrench_conv(wrench).unsqueeze(3).unsqueeze(3)
        gout = self.gripper_conv(gripper_state).unsqueeze(3).unsqueeze(3)
        feat = torch.cat((i3d_feat, wout, gout), axis=1)
        per_clip_logits = self.i3d.logits(self.i3d.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        return per_clip_logits

    def loss_function(self, output, batch):
        inputs, wrench, gripper_state, labels, vidx, trial_action = batch
        loss = F.cross_entropy(output, labels)
        return loss

