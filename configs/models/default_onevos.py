class ModelConfig():
    def __init__(self):
        self.MODEL_NAME = 'OneVOS'

        self.MODEL_VOS = 'aot_convmae'
        self.MODEL_ENGINE = 'aotengine_convmae'

        self.expand_ratio_small=1.3
        print("small obj expand ration:",self.expand_ratio_small)

        self.MODEL_ALIGN_CORNERS = False
        self.MODEL_ENCODER = 'mobilenetv2'
        self.MODEL_ENCODER_PRETRAIN = "/mnt/liwy/pretrain_ckpt/convmae/convmae_base.pth"
        # self.MODEL_ENCODER_PRETRAIN = ""
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]  # 4x, 8x, 16x, 16x
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 10
        self.MODEL_SELF_HEADS = 8
        self.MODEL_ATT_HEADS = 8
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_USE_PREV_PROB = False

        self.TRAIN_LONG_TERM_MEM_GAP = 9999

        self.TEST_LONG_TERM_MEM_GAP = 9999
