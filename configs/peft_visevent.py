import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PEFT_VISEVENT'

        self.init_dir()

        self.DATASETS = ['visevent',]
        self.DATA_SEQ_LEN = 5

        self.TRAIN_LR = 1e-4
        self.TRAIN_LR_MIN = 1e-5
        self.TRAIN_TOTAL_STEPS = 20000

        # for gradient accumulation
        self.ACCUMULATION_STEPS = 1
        self.TRAIN_MEM_EVERY=1
        self.training =True
        print("train mem every:",self.TRAIN_MEM_EVERY)

        self.PRETRAIN = True
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL ="weights/onevos_SDYM.pth"
