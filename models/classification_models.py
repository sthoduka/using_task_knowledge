import torch.nn as nn
import torch

from torchvision import models
import torch.nn.functional as F

from slowfast.models import build_model
from slowfast.utils.parser import load_config
import argparse

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
        out = self.video_model(vid) # TODO: remove temp_emb from MVIT model
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
