from mmcv.runner import BaseModule
from mmdet3d.models import BACKBONES

try:
    import timm
except ImportError:
    timm = None


@BACKBONES.register_module()
class TimmFeatureBackbone(BaseModule):
    def __init__(self,
                 model_name,
                 pretrained=True,
                 out_indices=(2, 3),
                 in_chans=3,
                 freeze=False,
                 init_cfg=None,
                 **kwargs):
        super(TimmFeatureBackbone, self).__init__(init_cfg)
        if timm is None:
            raise ImportError('timm is required to use TimmFeatureBackbone')

        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.freeze = freeze
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
            **kwargs)

        if self.freeze:
            self._freeze_backbone()

    def _freeze_backbone(self):
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super(TimmFeatureBackbone, self).train(mode)
        if self.freeze:
            self._freeze_backbone()
        return self

    def forward(self, x):
        return self.backbone(x)