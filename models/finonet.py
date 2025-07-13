####################
# The contents of this file are copied from https://github.com/ardai/fino-net/blob/main/src/finonet-rgb-d-a.ipynb
# with minor modifications
# The original code base has an MIT License
####################

import torch.nn as nn
import torch.nn.functional as F
import torchvision
try:
    from models.convlstm import ConvLSTM
except:
    from convlstm import ConvLSTM

import pdb

class VGGRGB(nn.Module):
    def __init__(self, hparams):
        super(VGGRGB, self).__init__()

        self.hparams = hparams
        self.vgg_model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        for params in self.vgg_model.parameters():
            params.requires_grad = False

        self.convlstm = ConvLSTM(input_dim=512,
                hidden_dim=[1024],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
                bias=True,
                return_all_layers=False)
        self.linear = nn.Linear(50176, self.hparams.num_outcome_classes)

    def forward(self, batch):
        x, robot_actions, label, _ = batch

        ### BLOCK 1 ###
        batch_size,frame_size,channel,height,width = x.size() # 8,4,3,224,224
        x_in1 = x.view(batch_size*frame_size, channel, height, width)
        x = self.vgg_model.features(x_in1) # 32 x 512 x 7 x 7
        x = x.view(batch_size, frame_size, 512, 7, 7)
        x = self.convlstm(x)[0][0] # 8 x 4 x 1024 x 7 x 7
        x = x[:,-1,:] # get last frame features
        x = x.view(batch_size, -1)
        x = self.linear(F.dropout(x, p=0.5))
        return x

    def loss_function(self, output, batch):
        x, robot_actions, label, _ = batch
        loss = F.binary_cross_entropy_with_logits(output, label)
        return loss

