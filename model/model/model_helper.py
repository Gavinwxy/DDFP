import torch.nn as nn
from .resnet import resnet101
from .decoder import dec_deeplabv3_plus


class ModelBuilder(nn.Module):
    def __init__(self, num_classes=21):
        super(ModelBuilder, self).__init__()

        self._num_classes = num_classes
        
        self.encoder = resnet101(multi_grid=False,
                                zero_init_residual=False,
                                fpn=True,
                                replace_stride_with_dilation=[False, False, True]
                                )
        
        self.decoder = dec_deeplabv3_plus(in_planes=self.encoder.get_outplanes(),
                                          num_classes=self._num_classes,
                                          inner_planes=256,
                                          dilations=[6, 12, 18])
        

    def forward(self, x, label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False):

        f1, f2, feat1, feat2 = self.encoder(x)
        outs = self.decoder([f1, f2, feat1, feat2], label, nf_model, loss_flow, cfg, eps, adv)
        
        return outs