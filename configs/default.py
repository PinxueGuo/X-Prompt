import os
import importlib


class DefaultEngineConfig():
    def __init__(self, exp_name='default', model='AOTT'):
        model_cfg = importlib.import_module('configs.models.' +
                                            model).ModelConfig()
        self.__dict__.update(model_cfg.__dict__)  # add model config

        backbone_cfg = importlib.import_module('configs.models.convmae.convmae').VitOnevosConfig()
        self.__dict__.update(backbone_cfg .__dict__) # add backbone config

        self.EXP_NAME = exp_name + '_' + self.MODEL_NAME

        self.STAGE_NAME = 'default'

        self.DATASETS = ['youtubevos']
        self.DATA_WORKERS = 8
        self.DATA_RANDOMCROP = (465,
                                465) if self.MODEL_ALIGN_CORNERS else (464,
                                                                       464)
        self.DATA_RANDOMFLIP = 0.5
        self.DATA_MAX_CROP_STEPS = 10
        self.DATA_SHORT_EDGE_LEN = 480
        self.DATA_MIN_SCALE_FACTOR = 0.7
        self.DATA_MAX_SCALE_FACTOR = 1.3
        self.DATA_RANDOM_REVERSE_SEQ = True
        self.DATA_SEQ_LEN = 5
        self.DATA_DAVIS_REPEAT = 5
        self.DATA_RANDOM_GAP_DAVIS = 12  # max frame interval between two sampled frames for DAVIS (24fps)
        self.DATA_RANDOM_GAP_YTB = 3  # max frame interval between two sampled frames for YouTube-VOS (6fps)
        self.DATA_DYNAMIC_MERGE_PROB = 0.3

        self.PRETRAIN = True
        self.PRETRAIN_FULL = False  # if False, load encoder only

        self.PRETRAIN_MODEL = ''

        self.TRAIN_TOTAL_STEPS = 100000
        self.TRAIN_START_STEP = 0
        self.TRAIN_WEIGHT_DECAY = 0.07
        self.TRAIN_WEIGHT_DECAY_EXCLUSIVE = {
            # 'encoder.': 0.01
        }
        self.TRAIN_WEIGHT_DECAY_EXEMPTION = [
            'absolute_pos_embed', 'relative_position_bias_table',
            'relative_emb_v', 'conv_out'
        ]
        self.TRAIN_LR = 2e-4
        self.TRAIN_LR_MIN = 2e-5 if 'mobilenetv2' in self.MODEL_ENCODER else 1e-5
        self.TRAIN_LR_POWER = 0.9
        self.TRAIN_LR_ENCODER_RATIO = 5.0
        self.TRAIN_LR_WARM_UP_RATIO = 0.05
        self.TRAIN_LR_COSINE_DECAY = False
        self.TRAIN_LR_RESTART = 1
        self.TRAIN_LR_UPDATE_STEP = 1
        self.TRAIN_AUX_LOSS_WEIGHT = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 1.0
        self.TRAIN_OPT = 'adamw'
        self.TRAIN_SGD_MOMENTUM = 0.9
        self.TRAIN_GPUS = 4
        self.TRAIN_BATCH_SIZE = 16
        self.TRAIN_TBLOG = True
        self.TRAIN_TBLOG_STEP = 50
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_IMG_LOG = True
        self.TRAIN_TOP_K_PERCENT_PIXELS = 0.15
        self.TRAIN_SEQ_TRAINING_FREEZE_PARAMS = ['patch_wise_id_bank']

        self.TRAIN_SEQ_TRAINING_START_RATIO = 0.5
        self.TRAIN_HARD_MINING_RATIO = 0.5
        self.TRAIN_EMA_RATIO = 0.1
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_SAVE_STEP = 2000
        self.TRAIN_MAX_KEEP_CKPT = 30
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_DATASET_FULL_RESOLUTION = False
        self.TRAIN_ENABLE_PREV_FRAME = False
        self.TRAIN_ID_DROPOUT = 0.

        self.TEST_GPU_ID = 0
        self.TEST_GPU_NUM = 1
        self.TEST_FRAME_LOG = False
        self.TEST_DATASET = 'youtubevos'
        self.TEST_DATASET_FULL_RESOLUTION = False
        self.TEST_DATASET_SPLIT = 'val'
        self.TEST_CKPT_PATH = None
        # if "None", evaluate the latest checkpoint.
        self.TEST_CKPT_STEP = None
        self.TEST_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MIN_SIZE = None
        self.TEST_MAX_SIZE = 800 * 1.3
        self.TEST_WORKERS = 2

        # GPU distribution
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "nccl"  # "gloo"
        self.DIST_URL = "tcp://127.0.0.1:13241"
        self.DIST_START_GPU = 0
        self.training=True
        self.TEST_PANO = False

        # MVP-VOS config
        self.EXPERT_LOW_RANK = 8
        self.EXPERT_NUM = 2


    def init_dir(self):
        self.DIR_DATA = '../datasets'
        self.DIR_DAVIS = os.path.join(self.DIR_DATA, 'DAVIS')
        self.DIR_YTB = os.path.join(self.DIR_DATA, 'YTB')
        self.DIR_STATIC = os.path.join(self.DIR_DATA, 'Static')
        self.DIR_DAVIS_eval = self.DIR_DAVIS
        self.DIR_YTB_eval = self.DIR_YTB

        self.DIR_VIST300 = os.path.join(self.DIR_DATA, 'VisT300')
        self.DIR_VTUVA = os.path.join(self.DIR_DATA, 'VT-UVA')
        self.DIR_ARKITTRACK = os.path.join(self.DIR_DATA, 'ARKitTrack')
        self.DIR_VISEVENT = os.path.join(self.DIR_DATA, 'VisEvent_new')

        self.DIR_ROOT = './'#'./data_wd/youtube_vos_jobs'
        self.DIR_RESULT = os.path.join(self.DIR_ROOT, 'outputs/train', self.EXP_NAME, self.STAGE_NAME)
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, 'ema_ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, 'log', 'tensorboard')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'img_logs')
        self.DIR_EVALUATION = './outputs/eval'

        for path in [
                self.DIR_RESULT, self.DIR_CKPT, self.DIR_EMA_CKPT,
                self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG,
                self.DIR_TB_LOG
        ]:
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except Exception as inst:
                    print(inst)
                    print('Failed to make dir: {}.'.format(path))

        # ### 172
        # self.DIR_DATA = "../DATA/"
        # self.DIR_DAVIS = "/mnt/share172/et21-guopx/codes/DATA/DAVIS/2017/trainval/"
        # self.DIR_YTB = "/home/et21-guopx/liwy-new/onevos/data/ytb/"
        # self.DIR_STATIC = "/mnt/share102/et21-guopx/lwy/code/liwy/DATA/AOT-static/VOS_Pretrain/"
        # self.DIR_ROOT = "/mnt/share102/et21-guopx/lwy/onevos_from0601/AOT/6.7 decoup new1/PRE_DAV_YTB_VIT/"
        # self.DIR_MOSE = "/mnt/share102/cs22-liwy/data/mose/"
        # self.DIR_DAVIS_eval = self.DIR_DAVIS
        # self.DIR_YTB_eval = "/home/et21-guopx/liwy-new/onevos/data/ytb/"
        # self.DIR_LLVOS = "/mnt/share102/cs22-liwy/LLVOS/VTRelease/"
        # self.DIR_VIP = "/mnt/share172/et21-guopx/lwy/dataset/VIPOSeg_valid/"

        # ### ???
        # self.DIR_STATIC = "/mnt/liwy/data/AOT-static/"
        # self.DIR_DAVIS = "/mnt/liwy/data/DAVIS/2017/trainval/"
        # self.DIR_YTB = "/mnt/liwy/data/AOT-YTB/"
        # self.DIR_DAVIS_eval = "/mnt/liwy/data/DAVIS/2017/test-dev/"
        # self.DIR_YTB_eval =self.DIR_YTB
        # self.DIR_MOSE = "/mnt/liwy/data/MOSE/"
        # self.DIR_LLVOS ="/mnt/liwy/data/LLVOS/VTRelease/"
        # self.DIR_Long_time_video ="/mnt/liwy/data/long-term-video-forval/"
        # self.DIR_VIP="/mnt/liwy/data/VIPOSeg_valid/"
        # self.DIR_LLVOS_Train = "/mnt/liwy/data/LLVOS/TrainGT/"
