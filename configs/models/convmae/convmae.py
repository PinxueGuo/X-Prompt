from easydict import EasyDict as edict

class VitOnevosConfig():
    def __init__(self):
        self.BACKBONE=edict()
        self.BACKBONE.VIT_TYPE = 'convmae_base'
        self.BACKBONE.PRETRAINED = True
        self.BACKBONE.PRETRAINED_PATH ="/mnt/share102/et21-guopx/lwy/pretrain_model_ckpt/convmae/convmae_base.pth"

        self.BACKBONE.EMBED_DIM = [256, 384, 768]
        #self.BACKBONE.patch_size=14
        self.BACKBONE.patch_size = 16
        self.BACKBONE.POSITION_EMBEDDING = 'sine'  # sine or learned


