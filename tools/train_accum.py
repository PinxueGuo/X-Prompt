import importlib
import random
import sys

sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')

import torch.multiprocessing as mp

from networks.managers.trainer import Trainer

#--pretrained_path
def main_worker(gpu, cfg, enable_amp=True):
    # Initiate a training manager
    trainer = Trainer(rank=gpu, cfg=cfg, enable_amp=enable_amp)
    # Start Training
    trainer.sequential_training()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VOS")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')

    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--dist_url', type=str, default='')
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--pretrained_path', type=str, default='')

    parser.add_argument('--datasets', nargs='+', type=str, default=[])
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1.)
    parser.add_argument('--start_step', type=int, default=-1.)

    args = parser.parse_args()

    engine_config = importlib.import_module('configs.' + args.stage)

    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    if len(args.datasets) > 0:
        cfg.DATASETS = args.datasets

    cfg.DIST_START_GPU = args.start_gpu
    if args.gpu_num > 0:
        cfg.TRAIN_GPUS = args.gpu_num
    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.pretrained_path != '':
        cfg.PRETRAIN_MODEL = args.pretrained_path

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr

    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step

    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step

    if args.dist_url == '':
        cfg.DIST_URL = 'tcp://127.0.0.1:123' + str(random.randint(0, 9)) + str(
            random.randint(0, 9))
    else:
        cfg.DIST_URL = args.dist_url

    cfg.is_topk=False

    # for record
    record_txt = cfg.DIR_RESULT + "/train_info_record.txt"
    cfg.IDEAL_BATCH_SIZE=cfg.TRAIN_BATCH_SIZE*cfg.ACCUMULATION_STEPS
    cfg.TRAIN_LOG_STEP = cfg.TRAIN_LOG_STEP * cfg.ACCUMULATION_STEPS
    cfg.is_topk=False
    with open(record_txt, mode='w+') as f:
        f.write("\n")
        f.write("DATA_SEQ_LEN= " + str(cfg.DATA_SEQ_LEN) + "\n")
        f.write("self.IDEAL_BATCH_SIZE= " + str(cfg.IDEAL_BATCH_SIZE) + "\n")
        f.write("self.TRAIN_BATCH_SIZE= " + str(cfg.TRAIN_BATCH_SIZE) + "\n")
        f.write("self.ACCUMULATION_STEPS= " + str(cfg.ACCUMULATION_STEPS) + "\n")
        f.write("self.TRAIN_LR= " + str(cfg.TRAIN_LR) + "\n")
        f.write("self.TRAIN_LR_MIN= " + str(cfg.TRAIN_LR_MIN) + "\n")
        f.write("self.TRAIN_TOTAL_STEPS= " + str(cfg.TRAIN_TOTAL_STEPS) + "\n")
        f.write("self.TRAIN_MEM_EVERT= " + str(cfg.TRAIN_MEM_EVERY) + "\n")
    # Use torch.multiprocessing.spawn to launch distributed processes
    mp.spawn(main_worker, nprocs=cfg.TRAIN_GPUS, args=(cfg, args.amp))


if __name__ == '__main__':
    main()
