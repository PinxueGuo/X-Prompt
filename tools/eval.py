import importlib
import sys

sys.path.append('.')
sys.path.append('..')

import torch
import torch.multiprocessing as mp

from networks.managers.evaluator_mem_new_ytb import Evaluator


def main_worker(gpu, cfg, seq_queue=None, info_queue=None, enable_amp=False, mem_every=20):
    # Initiate a evaluating manager
    evaluator = Evaluator(rank=gpu,
                          cfg=cfg,
                          seq_queue=seq_queue,
                          info_queue=info_queue,)
    # Start evaluation
    if enable_amp:
        with torch.cuda.amp.autocast(enabled=True):
            evaluator.evaluating(mem_every=mem_every)
            #evaluator.evaluating()
    else:
        evaluator.evaluating(mem_every=mem_every)
        #evaluator.evaluating()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eval VOS")
    parser.add_argument('--exp_name', type=str, default='default')

    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=1)

    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--split', type=str, default='')

    parser.add_argument('--no_ema', action='store_true')
    parser.set_defaults(no_ema=False)

    parser.add_argument('--flip', action='store_true')
    parser.set_defaults(flip=False)
    parser.add_argument('--ms', nargs='+', type=float, default=[1.])

    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--expert_num', type=int, default=2)

    args = parser.parse_args()

    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TEST_EMA = not args.no_ema
    cfg.TEST_EMA=False

    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_GPU_NUM = args.gpu_num

    if args.ckpt_path != '':
        cfg.TEST_CKPT_PATH = args.ckpt_path
    if args.ckpt_step > 0:
        cfg.TEST_CKPT_STEP = args.ckpt_step

    if args.dataset != '':
        cfg.TEST_DATASET = args.dataset

    if args.split != '':
        cfg.TEST_DATASET_SPLIT = args.split

    cfg.TEST_FLIP = args.flip
    cfg.TEST_MULTISCALE = args.ms

    cfg.TEST_MIN_SIZE = None
    cfg.TEST_MAX_SIZE = args.max_resolution * 800. / 480.

    cfg.EXPERT_NUM = args.expert_num

    if args.gpu_num > 1:
        mp.set_start_method('spawn')
        seq_queue = mp.Queue()
        info_queue = mp.Queue()

    ''' for common memory setting '''
    cfg.mem_capacity=10
    mem_every=5
    ''' for topk and DTS memory setting'''
    cfg.is_topk = False
    cfg.is_topk_percent = False
    # cfg.topk_tempalte = 0
    # cfg.topk_search = 0
    # cfg.topk_layer =[]
    # cfg.token_ratios=[0.39,1.00, 0.74, 1.00, 0.19, 0.00, 0.41, 1.00,0.00,0.09,0.09]
    ''' for small object eval '''
    cfg.larger_selection=True
    cfg.enter_small_obj=8
    cfg.expand_ratio_kernal=9


    print("ck:", cfg.TEST_CKPT_PATH)
    print("mem_every:", mem_every)
    print("mem_capcity:", cfg.mem_capacity)
    print("store:", cfg.DIR_EVALUATION)

    if args.gpu_num > 1:
        mp.spawn(main_worker,
                    nprocs=cfg.TEST_GPU_NUM,
                    args=(cfg, seq_queue, info_queue, args.amp,mem_every))
    else:
        main_worker(0, cfg, enable_amp=args.amp, mem_every=mem_every)


if __name__ == '__main__':
    main()
