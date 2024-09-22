import torch
import torch.nn as nn
from torch.nn import functional as F
from .attack import attack


class MixSyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixSyncBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixSyncBatchNorm, self).forward(input)
        elif self.batch_type == 'warm_up':
            batch_size = input.shape[0]
            input0 = super(MixSyncBatchNorm, self).forward(input[:batch_size//2])
            input1 = self.aux_bn(input[batch_size//2:])   
            input = torch.cat((input0, input1), 0)
        else:
            # In setting of tri, we have labeled features, strong aug features, pt features, three sets
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            clean_bd = batch_size // 3 * 2
            input0 = super(MixSyncBatchNorm, self).forward(input[:clean_bd])
            input1 = self.aux_bn(input[clean_bd:])   
            input = torch.cat((input0, input1), 0)

        return input


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)
    ):
        super(ASPP, self).__init__()

        norm_layer = nn.SyncBatchNorm
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = nn.SyncBatchNorm
        mix_norm = MixSyncBatchNorm

        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1), norm_layer(48), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=True),
            mix_norm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            mix_norm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.final = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, label, nf_model, loss_flow, cfg, eps, adv=False):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )

        fts_aspp = torch.cat((low_feat, aspp_out), dim=1)
        fts_aspp_pt = fts_aspp.detach()

        batch_num = fts_aspp_pt.size(0)
        if adv:
            fts_half_pt = fts_aspp_pt[batch_num // 2 : ]
            pt = attack(fts_half_pt, label, self.classifier, self.final, nf_model, loss_flow, cfg, eps)
            fts_half_pt = fts_aspp[batch_num // 2 : ].clone() + pt
            fts_aspp = torch.cat((fts_aspp, fts_half_pt), dim=0)
        
        fts = self.classifier(fts_aspp)
        fts_clean = fts[:batch_num].detach()
        pred = self.final(fts)
        
        return {'pred': pred, 'fts': fts_clean}


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19):
        super(Aux_Module, self).__init__()

        norm_layer = nn.SyncBatchNorm
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res
