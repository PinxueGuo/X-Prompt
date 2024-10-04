from .default import DefaultEngineConfig
import os


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PRE'

        self.init_dir()

        self.DATASETS = ['static']
        self.DATA_SEQ_LEN = 4

        self.DATA_DYNAMIC_MERGE_PROB = 1.0

        '''self.TRAIN_LR = 1e-4
        self.TRAIN_LR_MIN = 1e-5'''
        self.TRAIN_LR = 2e-4
        self.TRAIN_LR_MIN = 2e-5
        self.TRAIN_TOTAL_STEPS = 200000
        self.TRAIN_WEIGHT_DECAY = 0.03
        self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 0.1
        self.DATA_WORKERS = 8

        self.ACCUMULATION_STEPS = 2
        self.TRAIN_MEM_EVERY = 1
        self.PRETRAIN = False
        #self.block_selection=[2,5,8]
        #self.SPARSE_RATIO=0.8
        #self.block_keep_ratio = [self.SPARSE_RATIO, self.SPARSE_RATIO ** 2, self.SPARSE_RATIO ** 3]
