
from networks.decoders.fpn_convmae import FPNSegmentationHead_ori

def build_decoder(name, **kwargs):

    if name=='fpn_vitshortcut_deconv_convmae':
        return FPNSegmentationHead_ori(**kwargs)
    else:
        raise NotImplementedError