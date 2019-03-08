import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

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

class SSD300(nn.Module):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        vggt: pretrained vgg16 (partial) model
        label_num: number of classes (including background 0)
    """
    def __init__(self, label_num, backbone='resnet34', model_path="./resnet34-333f7ec4.pth"):

        super(SSD300, self).__init__()

        self.label_num = label_num

        if backbone == 'resnet34':
            self.model = ResNet34()
            out_channels = 256
            out_size = 38
            self.out_chan = [out_channels, 512, 512, 256, 256, 256]
        else:
            raise ValueError('Invalid backbone chosen')

        self._build_additional_features(out_size, self.out_chan)

        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.out_chan):
            self.loc.append(nn.Conv2d(oc, nd*4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd*label_num, kernel_size=3, padding=1))


        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        # intitalize all weights
        self._init_weights()

    def _build_additional_features(self, input_size, input_channels):
        idx = 0
        if input_size == 38:
            idx = 0
        elif input_size == 19:
            idx = 1
        elif input_size == 10:
            idx = 2

        self.additional_blocks = []

        if input_size == 38:
            self.additional_blocks.append(nn.Sequential(
                nn.Conv2d(input_channels[idx], 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ))
            idx += 1

        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # Only necessary in VGG for now
        if input_size >= 19:
            # conv11_1, conv11_2
            self.additional_blocks.append(nn.Sequential(
                nn.Conv2d(input_channels[idx], 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, input_channels[idx+1], kernel_size=3),
                nn.ReLU(inplace=True),
            ))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):

        layers = [
            *self.additional_blocks,
            *self.loc, *self.conf]

        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, data):

        layers = self.model(data)

        # last result from network goes into additional blocks
        x = layers[-1]
        additional_results = []
        for i, l in enumerate(self.additional_blocks):
            x = l(x)
            additional_results.append(x)

        src = [*layers, *additional_results]
        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4

        locs, confs = self.bbox_view(src, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


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
