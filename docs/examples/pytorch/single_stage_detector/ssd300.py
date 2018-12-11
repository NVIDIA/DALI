import torch
import torch.nn as nn
from base_model import ResNet34
from mlperf_compliance import mlperf_log

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
            mlperf_log.ssd_print(key=mlperf_log.BACKBONE, value='resnet34')
            out_channels = 256
            out_size = 38
            self.out_chan = [out_channels, 512, 512, 256, 256, 256]
            mlperf_log.ssd_print(key=mlperf_log.LOC_CONF_OUT_CHANNELS,
                                 value=self.out_chan)

        else:
            raise ValueError('Invalid backbone chosen')

        self._build_additional_features(out_size, self.out_chan)

        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        mlperf_log.ssd_print(key=mlperf_log.NUM_DEFAULTS_PER_CELL,
                             value=self.num_defaults)
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
    