"""
    Load the vgg16 weight and save it to special file
"""

#from torchvision.models.vgg import vgg16
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from collections import OrderedDict

from torchvision.models.resnet import resnet18, resnet34, resnet50

def _ModifyConvStrideDilation(conv, stride=(1, 1), padding=None):
    conv.stride = stride

    if padding is not None:
        conv.padding = padding

def _ModifyBlock(block, bottleneck=False, **kwargs):
    for m in list(block.children()):
        if bottleneck:
           _ModifyConvStrideDilation(m.conv2, **kwargs)
        else:
           _ModifyConvStrideDilation(m.conv1, **kwargs)

        if m.downsample is not None:
            # need to make sure no padding for the 1x1 residual connection
            _ModifyConvStrideDilation(list(m.downsample.children())[0], **kwargs)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        rn18 = resnet18(pretrained=True)


        # discard last Resnet block, avrpooling and classification FC
        # layer1 = up to and including conv3 block
        self.layer1 = nn.Sequential(*list(rn18.children())[:6])
        # layer2 = conv4 block only
        self.layer2 = nn.Sequential(*list(rn18.children())[6:7])

        # modify conv4 if necessary
        # Always deal with stride in first block
        modulelist = list(self.layer2.children())
        _ModifyBlock(modulelist[0], stride=(1,1))

    def forward(self, data):
        layer1_activation = self.layer1(data)
        x = layer1_activation
        layer2_activation = self.layer2(x)

        # Only need the output of conv4
        return [layer2_activation]

class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        rn34 = resnet34(pretrained=True)

        # discard last Resnet block, avrpooling and classification FC
        self.layer1 = nn.Sequential(*list(rn34.children())[:6])
        self.layer2 = nn.Sequential(*list(rn34.children())[6:7])
        # modify conv4 if necessary
        # Always deal with stride in first block
        modulelist = list(self.layer2.children())
        _ModifyBlock(modulelist[0], stride=(1,1))


    def forward(self, data):
        layer1_activation = self.layer1(data)
        x = layer1_activation
        layer2_activation = self.layer2(x)

        return [layer2_activation]

class L2Norm(nn.Module):
    """
       Scale shall be learnable according to original paper
       scale: initial scale number
       chan_num: L2Norm channel number (norm over all channels)
    """
    def __init__(self, scale=20, chan_num=512):
        super(L2Norm, self).__init__()
        # Scale across channels
        self.scale = \
            nn.Parameter(torch.Tensor([scale]*chan_num).view(1, chan_num, 1, 1))

    def forward(self, data):
        # normalize accross channel
        return self.scale*data*data.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-12).rsqrt()



def tailor_module(src_model, src_dir, tgt_model, tgt_dir):
    state = torch.load(src_dir)
    src_model.load_state_dict(state)
    src_state = src_model.state_dict()
    # only need features
    keys1 = src_state.keys()
    keys1 = [k for k in src_state.keys() if k.startswith("features")]
    keys2 = tgt_model.state_dict().keys()

    assert len(keys1) == len(keys2)
    state = OrderedDict()

    for k1, k2 in zip(keys1, keys2):
        # print(k1, k2)
        state[k2] = src_state[k1]
    #diff_keys = state.keys() - target_model.state_dict().keys()
    #print("Different Keys:", diff_keys)
    # Remove unecessary keys
    #for k in diff_keys:
    #    state.pop(k)
    tgt_model.load_state_dict(state)
    torch.save(tgt_model.state_dict(), tgt_dir)

# Default vgg16 in pytorch seems different from ssd
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # Notice ceil_mode is true
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()

        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """

        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con*(mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)

        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret

