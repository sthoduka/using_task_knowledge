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
        elif 'handover' in self.hparams.dataset:
            img_pair, robot_action, lbl, trial_id, orig_lbl = batch
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
        elif 'handover' in self.hparams.dataset:
            img_pair, robot_action, lbl, trial_id, orig_lbl = batch
            loss = F.cross_entropy(output, lbl)
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
        if 'handover' in hparams.dataset or ('vtd' in hparams.dataset and 'v' in hparams.vtd_data_type):
            if hparams.video_type == 'rgb':
                self.i3d = InceptionI3d(400, in_channels=3)
            else:
                self.i3d = InceptionI3d(400, in_channels=2)
            if load_pretrained:
                if hparams.i3d_model_path != '':
                    print('Loading %s weights ' % hparams.i3d_model_path)
                    self.i3d.load_state_dict(torch.load(hparams.i3d_model_path))
        in_channels = 1024
        if 'handover' in hparams.dataset:
            in_channels = in_channels + 16 + 16
        if 'vtd' in hparams.dataset:
            if 'v' not in hparams.vtd_data_type:
                in_channels = 0
            if 't' in hparams.vtd_data_type:
                in_channels += 16
            if 'p' in hparams.vtd_data_type:
                in_channels += 16

        if 'vtd' in hparams.dataset and 'v' not in hparams.vtd_data_type:
            self.logits = Unit3D(in_channels=in_channels, output_channels=hparams.num_classes,
                                 kernel_shape=[1, 1, 1],
                                 padding=0,
                                 activation_fn=None,
                                 use_batch_norm=False,
                                 use_bias=True,
                                 name='logits')
        else:
            self.i3d.replace_logits(hparams.num_classes, in_channels=in_channels)

        if 'handover' in hparams.dataset:
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
        if 'vtd' in hparams.dataset:
            if 't' in hparams.vtd_data_type:
                if hparams.vtd_tactile_data_type == 'image':
                    self.tactile_model_conv = nn.Sequential(
                            nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=2),
                            nn.BatchNorm3d(32),
                            nn.ReLU(),
                            nn.Conv3d(32, 16, kernel_size=(3,3,3), padding=0),
                            nn.BatchNorm3d(16),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(3,3,3)),
                            nn.AdaptiveAvgPool3d((7, 1, 1)),
                    )
                else:
                    self.tactile_model_conv = nn.Sequential(
                            nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2),
                            nn.BatchNorm1d(32),
                            nn.ReLU(),
                            nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool1d(7),
                    )
            if 'p' in hparams.vtd_data_type:
                if hparams.vtd_tactile_data_type == 'image':
                    self.position_model_conv = nn.Sequential(
                            nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=2),
                            nn.BatchNorm3d(32),
                            nn.ReLU(),
                            nn.Conv3d(32, 16, kernel_size=(3,3,3), padding=0),
                            nn.BatchNorm3d(16),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(3,3,3)),
                            nn.AdaptiveAvgPool3d((7, 1, 1)),
                    )
                else:
                    self.position_model_conv = nn.Sequential(
                            nn.Conv1d(8, 32, kernel_size=3, padding=2, dilation=2),
                            nn.BatchNorm1d(32),
                            nn.ReLU(),
                            nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool1d(7),
                    )
        self.hparams = hparams

    def forward(self, batch):
        if 'handover' in self.hparams.dataset:
            inputs, wrench, gripper_state, labels, vidx, trial_action = batch
        elif 'vtd' in self.hparams.dataset:
            inputs, joint_pos, tactile, labels, vid = batch

        if 'handover' in self.hparams.dataset:
            i3d_feat = self.i3d.extract_features(inputs)
            wout = self.wrench_conv(wrench).unsqueeze(3).unsqueeze(3)
            gout = self.gripper_conv(gripper_state).unsqueeze(3).unsqueeze(3)
            feat = torch.cat((i3d_feat, wout, gout), axis=1)
            per_clip_logits = self.i3d.logits(self.i3d.dropout(feat)).squeeze(3).squeeze(3)
            per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        elif 'vtd' in self.hparams.dataset:
            if 'v' in self.hparams.vtd_data_type:
                i3d_feat = self.i3d.extract_features(inputs)
                feat = i3d_feat
            if 't' in self.hparams.vtd_data_type:
                tactile_feat = self.tactile_model_conv(tactile)
                if tactile_feat.dim() == 3:
                    tactile_feat = tactile_feat.unsqueeze(3).unsqueeze(3)
                if 'v' in self.hparams.vtd_data_type:
                    feat = torch.cat((feat, tactile_feat), axis=1)
                else:
                    feat = tactile_feat
            if 'p' in self.hparams.vtd_data_type:
                pos_feat = self.position_model_conv(joint_pos)
                if pos_feat.dim() == 3:
                    pos_feat = pos_feat.unsqueeze(3).unsqueeze(3)
                if 'v' in self.hparams.vtd_data_type or 't' in self.hparams.vtd_data_type:
                    feat = torch.cat((feat, pos_feat), axis=1)
                else:
                    feat = pos_feat
            if 'v' in self.hparams.vtd_data_type:
                per_clip_logits = self.i3d.logits(self.i3d.dropout(feat)).squeeze(3).squeeze(3)
                per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
            else:
                per_clip_logits = self.logits(F.dropout(feat)).squeeze(3).squeeze(3)
                per_clip_logits = torch.max(per_clip_logits, dim=2)[0]

        return per_clip_logits

    def loss_function(self, output, batch):
        if 'handover' in self.hparams.dataset:
            inputs, wrench, gripper_state, labels, vidx, trial_action = batch
        elif 'vtd' in self.hparams.dataset:
            inputs, joint_pos, tactile, labels, vid = batch
        loss = F.cross_entropy(output, labels)
        return loss

