import math
import os
import copy
import json
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
from torchvision import models
import torchvision
import pytorch_lightning as pl

from datasets.armbench_dataset import get_armbench_dataset, accumulate_armbench_results, compute_armbench_metrics
from datasets.failure_dataset import get_failure_dataset, accumulate_failure_results, compute_failure_metrics, accumulate_failure_video_results
from datasets.imperfect_pour_dataset import get_imp_dataset, accumulate_imp_results, compute_imp_metrics

from models.classification_models import VideoClassifier, ImgPairClassifier
from models.finonet import VGGRGB

import pdb

class FailureClassificationTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.dataset == 'armbench_video' or self.hparams.dataset == "armbench_image_video_mvit":
            if self.hparams.finonet:
                self.model = VGGRGB(self.hparams)
            else:
                self.model = VideoClassifier(self.hparams)
                if self.hparams.partial_ckpt != '':
                    self.load_state_dict(torch.load(self.hparams.partial_ckpt)['state_dict'], strict=False)
        elif self.hparams.dataset == 'armbench_img_pair':
            self.model = ImgPairClassifier(self.hparams)
        elif self.hparams.dataset == 'imperfect_pour':
            self.model = VideoClassifier(self.hparams)
        elif self.hparams.dataset == 'imperfect_pour_img_pair':
            self.model = ImgPairClassifier(self.hparams)
        elif self.hparams.dataset == 'failure_video':
            self.model = VideoClassifier(self.hparams)
        elif self.hparams.dataset == 'failure_img_pair':
            self.model = ImgPairClassifier(self.hparams)
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, batch):
        out = self.model(batch)
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.model.loss_function(out, batch)
        self.train_outputs.append(loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.model.loss_function(out, batch)
        if 'armbench' in self.hparams.dataset:
            outputs = accumulate_armbench_results(batch, out, self.hparams)
        elif 'imperfect_pour' in self.hparams.dataset:
            outputs = accumulate_imp_results(batch, out, self.hparams)
        elif self.hparams.dataset == 'failure_video':
            outputs = accumulate_failure_video_results(batch, out, self.hparams)
        elif 'failure_img_pair' in self.hparams.dataset:
            outputs = accumulate_failure_video_results(batch, out, self.hparams)
        outputs['loss'] = loss
        self.val_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.model.loss_function(out, batch)
        if 'armbench' in self.hparams.dataset:
            outputs = accumulate_armbench_results(batch, out, self.hparams)
        elif 'imperfect_pour' in self.hparams.dataset:
            outputs = accumulate_imp_results(batch, out, self.hparams)
        elif self.hparams.dataset == 'failure_video':
            outputs = accumulate_failure_video_results(batch, out, self.hparams)
        elif 'failure_img_pair' in self.hparams.dataset:
            outputs = accumulate_failure_video_results(batch, out, self.hparams)
        outputs['loss'] = loss
        self.test_outputs.append(outputs)
        return outputs

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_outputs = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.val_outputs]).mean()
        self.log("val_loss", avg_loss)

        if 'armbench' in self.hparams.dataset:
            results, gt, predictions, trials, logits, robot_actions = compute_armbench_metrics(self.val_outputs, result_type='val')
        elif 'imperfect_pour' in self.hparams.dataset:
            results, gt, predictions, trials, logits = compute_imp_metrics(self.val_outputs, result_type='val')
        elif 'failure' in self.hparams.dataset:
            results, gt, predictions, logits, trial_names = compute_failure_metrics(self.val_outputs, result_type='val')

        for key, value in results.items():
            self.log(key, value)

        self.val_outputs.clear()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_outputs = []

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        self.log("test_loss", avg_loss)

        if 'armbench' in self.hparams.dataset:
            results, gt, pred, trial_names, logits, robot_actions = compute_armbench_metrics(self.test_outputs, result_type='test')
            if gt is not None:
                np.save(os.path.join(self.logger.log_dir, 'gt.npy'), gt.numpy())
                np.save(os.path.join(self.logger.log_dir, 'pred.npy'), pred.numpy())
                np.save(os.path.join(self.logger.log_dir, 'logits.npy'), logits.numpy())
                np.save(os.path.join(self.logger.log_dir, 'robot_actions.npy'), robot_actions)
                with open(os.path.join(self.logger.log_dir, 'trials.json'), 'w') as fp:
                    json.dump(trial_names, fp)
        elif 'imperfect_pour' in self.hparams.dataset:
            results, gt, pred, trial_names, logits = compute_imp_metrics(self.test_outputs, result_type='test')
            if self.hparams.selected_action != '':
                os.makedirs(os.path.join(self.logger.log_dir, self.hparams.selected_action))
                save_path = os.path.join(self.logger.log_dir, self.hparams.selected_action)
            else:
                save_path = self.logger.log_dir
            np.save(os.path.join(save_path, 'gt.npy'), gt.numpy())
            np.save(os.path.join(save_path, 'pred.npy'), pred.numpy())
            np.save(os.path.join(save_path, 'logits.npy'), logits.numpy())
            with open(os.path.join(save_path, 'trials.json'), 'w') as fp:
                json.dump(trial_names, fp)
        elif 'failure' in self.hparams.dataset:
            results, gt, pred, logits, trial_names = compute_failure_metrics(self.test_outputs, result_type='test')
            if self.hparams.selected_action != '':
                os.makedirs(os.path.join(self.logger.log_dir, self.hparams.selected_action))
                save_path = os.path.join(self.logger.log_dir, self.hparams.selected_action)
            else:
                save_path = self.logger.log_dir
            if gt is not None:
                np.save(os.path.join(save_path, 'gt.npy'), gt)
                np.save(os.path.join(save_path, 'pred.npy'), pred)
                np.save(os.path.join(save_path, 'logits.npy'), logits)
                with open(os.path.join(save_path, 'trials.json'), 'w') as fp:
                    json.dump(trial_names, fp)
        for key, value in results.items():
            if key not in ['test_gt', 'test_logits', 'test_predictions', 'test_def_seg']:
                self.log(key, value)
        self.test_outputs.clear()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_outputs = []

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_outputs).mean()
        self.log("train_loss", avg_loss)
        self.train_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.learning_rate/20)
        elif self.hparams.lr_scheduler == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.hparams.lr_step_size], gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        if 'armbench' in self.hparams.dataset:
            train_dataset = get_armbench_dataset(self.hparams, dataset_type='train')
        elif 'imperfect_pour' in self.hparams.dataset:
            train_dataset = get_imp_dataset(self.hparams, dataset_type='train')
        elif 'failure' in self.hparams.dataset:
            train_dataset = get_failure_dataset(self.hparams, dataset_type='train')
        if self.training:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=True, num_workers=self.hparams.n_threads, pin_memory=False)
        else:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

    def val_dataloader(self):
        if 'armbench' in self.hparams.dataset:
            val_dataset = get_armbench_dataset(self.hparams, dataset_type='val')
        elif 'imperfect_pour' in self.hparams.dataset:
            val_dataset = get_imp_dataset(self.hparams, dataset_type='val')
        elif 'failure' in self.hparams.dataset:
            val_dataset = get_failure_dataset(self.hparams, dataset_type='val')
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)
    def test_dataloader(self):
        if 'armbench' in self.hparams.dataset:
            test_dataset = get_armbench_dataset(self.hparams, dataset_type='test')
        elif 'imperfect_pour' in self.hparams.dataset:
            test_dataset = get_imp_dataset(self.hparams, dataset_type='test')
        elif 'failure' in self.hparams.dataset:
            test_dataset = get_failure_dataset(self.hparams, dataset_type='test')
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

