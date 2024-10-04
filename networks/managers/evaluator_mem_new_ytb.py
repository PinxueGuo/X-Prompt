import os
import time
import datetime as datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.eval_datasets import YOUTUBEVOS_Test, YOUTUBEVOS_DenseTest, DAVIS_Test, EVAL_TEST, MOSE_Test, LLVOS_Test, VIPOSeg_Test
from dataloaders.eval_datasets import VisT300_Test, VTUVA_Test, ARKitTrack_Test, VisEvent_Test
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder,zip_folder_mose

from networks.models import build_vos_model
from networks.engines import build_engine


class Evaluator(object):
    def __init__(self, cfg, rank=0, seq_queue=None, info_queue=None):
        self.gpu = cfg.TEST_GPU_ID + rank
        self.gpu_num = cfg.TEST_GPU_NUM
        self.rank = rank
        self.cfg = cfg
        self.seq_queue = seq_queue
        self.info_queue = info_queue

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

        print("Use GPU {} for evaluating.".format(self.gpu))
        torch.cuda.set_device(self.gpu)

        self.print_log('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg

        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return

        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(
                        map(lambda x: int(x.split('_')[-1].split('.')[0]),
                            ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            if cfg.TEST_EMA:
                cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT,
                                              'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu,
                                                    self.rank)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu,
                                                    self.rank)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                 cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                 cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        eval_name = '{}_{}_{}_{}_ckpt_{}'.format(cfg.TEST_DATASET,
                                                 cfg.TEST_DATASET_SPLIT,
                                                 cfg.EXP_NAME, cfg.STAGE_NAME,
                                                 self.ckpt)
        #cfg.TEST_EMA=False
        if cfg.TEST_EMA:
            eval_name += '_ema'
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms_' + str(cfg.TEST_MULTISCALE).replace(
                '.', 'dot').replace('[', '').replace(']', '').replace(
                    ', ', '_')

        if 'youtubevos' in cfg.TEST_DATASET:
            year = int(cfg.TEST_DATASET[-4:])
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            if '_all_frames' in cfg.TEST_DATASET_SPLIT:
                split = cfg.TEST_DATASET_SPLIT.split('_')[0]
                youtubevos_test = YOUTUBEVOS_DenseTest

                self.result_root_sparse = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_sparse',
                                                       'Annotations')
                self.zip_dir_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    '{}_sparse.zip'.format(eval_name))
                print("self.zip_dir_sparse:",self.zip_dir_sparse)
            else:
                split = cfg.TEST_DATASET_SPLIT
                youtubevos_test = YOUTUBEVOS_Test

            self.dataset = youtubevos_test(root=cfg.DIR_YTB_eval,
                                           year=year,
                                           split=split,
                                           transform=eval_transforms,
                                           result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS_eval,
                year=2017,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2016,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'mose':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            mose_test = MOSE_Test

            self.dataset = mose_test(root=cfg.DIR_MOSE,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'llvos':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            llvos_Test= LLVOS_Test

            self.dataset = llvos_Test(root=cfg.DIR_LLVOS,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'long_time_video':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            llvos_Test= LLVOS_Test

            self.dataset = llvos_Test(root=cfg.DIR_Long_time_video,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'viposeg':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            if cfg.TEST_SAVE_PROB:
                self.result_root_prob = os.path.join(cfg.DIR_EVALUATION,
                                                     cfg.TEST_DATASET,
                                                     eval_name + '_prob',
                                                     'Annotations')
            split = cfg.TEST_DATASET_SPLIT
            self.dataset = VIPOSeg_Test(root=cfg.DIR_VIP,
                                        split=split,
                                        transform=eval_transforms,
                                        result_root=self.result_root,
                                        test_pano=cfg.TEST_PANO)

        elif cfg.TEST_DATASET == 'vist300':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            vist300_test = VisT300_Test

            self.dataset = vist300_test(root=cfg.DIR_VIST300,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'vtuva':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            vtuva_test = VTUVA_Test

            self.dataset = vtuva_test(root=cfg.DIR_VTUVA,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'arkittrack':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            arkittrack_test = ARKitTrack_Test

            self.dataset = arkittrack_test(root=cfg.DIR_ARKITTRACK,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'visevent':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET,eval_name)
            split = cfg.TEST_DATASET_SPLIT
            visevent_test = VisEvent_Test

            self.dataset = visevent_test(root=cfg.DIR_VISEVENT,
                                            split=split,
                                            transform=eval_transforms,
                                            result_root=self.result_root)

        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)
        else:
            self.print_log('Unknown dataset!')
            exit()

        self.print_log('Eval {} on {} {}:'.format(cfg.EXP_NAME,
                                                  cfg.TEST_DATASET,
                                                  cfg.TEST_DATASET_SPLIT))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                          eval_name, 'Annotations')

        if cfg.TEST_DATASET == 'mose':
            self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                              eval_name)

        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                    '{}.zip'.format(eval_name))

        if not os.path.exists(self.result_root):
            try:
                os.makedirs(self.result_root)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Failed to mask dir: {}.'.format(
                    self.result_root))
        self.print_log('Done!')

    def evaluating(self,mem_every=10):
        print("now mem_every:",mem_every)
        cfg = self.cfg
        self.model.eval()
        video_num = 0
        processed_video_num = 0
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        start_eval_time = time.time()

        if self.seq_queue is not None:
            if self.rank == 0:
                for seq_idx in range(total_video_num):
                    self.seq_queue.put(seq_idx)
                for _ in range(self.gpu_num):
                    self.seq_queue.put('END')
            coming_seq_idx = self.seq_queue.get()

        all_engines = []
        with torch.no_grad():
            for seq_idx, seq_dataset in enumerate(self.dataset):
                print("seq:",seq_dataset.seq_name)
                torch.cuda.empty_cache()
                video_num += 1

                if self.seq_queue is not None:
                    if coming_seq_idx == 'END':
                        break
                    elif coming_seq_idx != seq_idx:
                        continue
                    else:
                        coming_seq_idx = self.seq_queue.get()

                processed_video_num += 1

                for engine in all_engines:
                    engine.restart_engine()

                seq_name = seq_dataset.seq_name
                print('GPU {} - Processing Seq {} [{}/{}]:'.format(
                    self.gpu, seq_name, video_num, total_video_num))
                #torch.cuda.empty_cache()

                seq_dataloader = DataLoader(seq_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=cfg.TEST_WORKERS,
                                            pin_memory=True)

                if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                    images_sparse = seq_dataset.images_sparse
                    seq_dir_sparse = os.path.join(self.result_root_sparse,
                                                  seq_name)
                    if not os.path.exists(seq_dir_sparse):
                        os.makedirs(seq_dir_sparse)

                seq_total_time = 0
                seq_total_frame = 0
                seq_pred_masks = {'dense': [], 'sparse': []}
                seq_timers = []
                i=0
                self.update_new_obj=False

                for frame_idx, samples in enumerate(seq_dataloader):
                    torch.cuda.empty_cache()

                    all_preds = []
                    new_obj_label = None

                    for aug_idx in range(len(samples)):
                        if len(all_engines) <= aug_idx:
                            all_engines.append(
                                build_engine(cfg.MODEL_ENGINE,
                                             phase='eval',
                                             aot_model=self.model,
                                             gpu_id=self.gpu,
                                             long_term_mem_gap=self.cfg.
                                             TEST_LONG_TERM_MEM_GAP,seq_name=seq_name))
                            all_engines[-1].eval()

                        engine = all_engines[aug_idx]

                        for engine in all_engines:
                            engine.seq_name=seq_name

                        sample = samples[aug_idx]

                        is_flipped = sample['meta']['flip']

                        obj_nums = sample['meta']['obj_num']
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        obj_idx = sample['meta']['obj_idx']

                        obj_nums = [int(obj_num) for obj_num in obj_nums]
                        obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                        current_img = sample['current_img']
                        current_img = current_img.cuda(self.gpu, non_blocking=True)
                        sample['current_img'] = current_img

                        current_aux_img = sample['current_aux_img']
                        current_aux_img = current_aux_img.cuda(self.gpu, non_blocking=True)
                        sample['current_aux_img'] = current_aux_img

                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(self.gpu, non_blocking=True).float()
                        else:
                            current_label = None

                        #############################################################

                        if frame_idx == 0:
                            if current_label is None:
                                B, _, H, W = current_img.shape
                                current_label = torch.zeros(B, 1, H, W).cuda(self.gpu, non_blocking=True).float()

                            _current_label = F.interpolate(current_label,
                                                           size=current_img.size()[2:],
                                                           mode="nearest")
                            # engine.process_reference_frame((current_img+current_aux_img)/2.,
                            engine.process_reference_frame(torch.cat([current_img,current_aux_img],dim=1),
                                                           _current_label,
                                                           frame_step=0,
                                                           obj_nums=obj_nums)
                            # self.ref_img=current_img
                            self.prev_img = current_img
                            self.prev_aux_img = current_aux_img
                            self.ref_obj = True
                        else:
                            if aug_idx == 0:
                                seq_timers.append([])
                                now_timer = torch.cuda.Event(enable_timing=True)
                                now_timer.record()
                                seq_timers[-1].append((now_timer))

                            # engine.match_propogate_one_frame((current_img+current_aux_img)/2., (self.prev_img+self.prev_aux_img)/2.)
                            engine.match_propogate_one_frame(torch.cat([current_img,current_aux_img],dim=1), torch.cat([self.prev_img,self.prev_aux_img],dim=1))
                            if self.ref_obj:
                                engine.update_mem()
                                self.ref_obj=False
                            elif self.update_new_obj:
                                engine.add_ref_num()
                                engine.update_mem()
                                self.update_new_obj = False
                            elif (frame_idx-1) % mem_every==0:
                                #print("mem update frame id:",frame_idx)
                                engine.update_mem()

                            self.prev_img = current_img
                            self.prev_aux_img = current_aux_img


                            pred_logit = engine.decode_current_logits((ori_height, ori_width))

                            if is_flipped:
                                pred_logit = flip_tensor(pred_logit, 3)

                            pred_prob = torch.softmax(pred_logit, dim=1)
                            all_preds.append(pred_prob)

                            if not is_flipped and current_label is not None and new_obj_label is None:
                                new_obj_label = current_label

                    if frame_idx > 0:
                        all_preds = torch.cat(all_preds, dim=0)
                        pred_prob = torch.mean(all_preds, dim=0, keepdim=True)
                        pred_label = torch.argmax(pred_prob,
                                                  dim=1,
                                                  keepdim=True).float()

                        if new_obj_label is not None:
                            keep = (new_obj_label == 0).float()
                            pred_label = pred_label * keep + new_obj_label * (1 - keep)
                            new_obj_nums = [int(pred_label.max().item())]

                            if cfg.TEST_FLIP:
                                flip_pred_label = flip_tensor(pred_label, 3)

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_img = samples[aug_idx]['current_img']
                                current_aux_img = samples[aug_idx]['current_aux_img']

                                current_label = flip_pred_label if samples[aug_idx]['meta']['flip'] else pred_label
                                current_label = F.interpolate(current_label,
                                                              size=engine.input_size_2d,
                                                              mode="nearest")
                                engine.update_prev_frame(current_label)
                                i+=1
                                print("enter: "+str(i))
                                self.update_new_obj=True

                        else:
                            if not cfg.MODEL_USE_PREV_PROB:
                                if cfg.TEST_FLIP:
                                    flip_pred_label = flip_tensor(pred_label, 3)

                                for aug_idx in range(len(samples)):
                                    engine = all_engines[aug_idx]
                                    current_label = flip_pred_label if samples[aug_idx]['meta']['flip'] else pred_label
                                    current_label = F.interpolate(current_label,
                                                                  size=engine.input_size_2d,
                                                                  mode="nearest")
                                    engine.update_prev_frame(current_label)
                            else:
                                if cfg.TEST_FLIP:
                                    flip_pred_prob = flip_tensor(pred_prob, 3)

                                for aug_idx in range(len(samples)):
                                    engine = all_engines[aug_idx]
                                    current_prob = flip_pred_prob if samples[aug_idx]['meta']['flip'] else pred_prob
                                    current_prob = F.interpolate(current_prob,
                                                                 size=engine.input_size_2d,
                                                                 mode="nearest")

                        now_timer = torch.cuda.Event(enable_timing=True)
                        now_timer.record()
                        seq_timers[-1].append((now_timer))

                        if cfg.TEST_FRAME_LOG:
                            torch.cuda.synchronize()
                            one_frametime = seq_timers[-1][0].elapsed_time(
                                seq_timers[-1][1]) / 1e3
                            obj_num = obj_nums[0]
                            print(
                                'GPU {} - Frame: {} - Obj Num: {}, Time: {}ms'.
                                format(self.gpu, '.'.join(imgname[0].split('.')[:-1]),
                                       obj_num, int(one_frametime * 1e3)))

                        # Save result
                        seq_pred_masks['dense'].append({
                            'path':
                            os.path.join(self.result_root, seq_name,
                                         '.'.join(imgname[0].split('.')[:-1]) + '.png'),
                            'mask':
                            pred_label,
                            'obj_idx':
                            obj_idx
                        })
                        if 'all_frames' in cfg.TEST_DATASET_SPLIT and imgname in images_sparse:
                            seq_pred_masks['sparse'].append({
                                'path':
                                os.path.join(self.result_root_sparse, seq_name,
                                             '.'.join(imgname[0].split('.')[:-1]) + '.png'),
                                'mask':
                                pred_label,
                                'obj_idx':
                                obj_idx
                            })

                # Save result
                for mask_result in seq_pred_masks['dense'] + seq_pred_masks[
                        'sparse']:
                    save_mask(mask_result['mask'].squeeze(0).squeeze(0),
                              mask_result['path'], mask_result['obj_idx'])
                del (seq_pred_masks)

                for timer in seq_timers:
                    torch.cuda.synchronize()
                    one_frametime = timer[0].elapsed_time(timer[1]) / 1e3
                    seq_total_time += one_frametime
                    seq_total_frame += 1
                del (seq_timers)

                seq_avg_time_per_frame = seq_total_time / seq_total_frame
                total_time += seq_total_time
                total_frame += seq_total_frame
                total_avg_time_per_frame = total_time / total_frame
                total_sfps += seq_avg_time_per_frame
                avg_sfps = total_sfps / processed_video_num
                max_mem = torch.cuda.max_memory_allocated(
                    device=self.gpu) / (1024.**3)
                max_mem_mb = torch.cuda.max_memory_allocated(
                    device=self.gpu) / (2**20)
                print(
                    "GPU {} - Seq {} - FPS: {:.2f}. All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                    .format(self.gpu, seq_name, 1. / seq_avg_time_per_frame,
                            1. / total_avg_time_per_frame, 1. / avg_sfps,
                            max_mem))


        if self.seq_queue is not None:
            if self.rank != 0:
                self.info_queue.put({
                    'total_time': total_time,
                    'total_frame': total_frame,
                    'total_sfps': total_sfps,
                    'processed_video_num': processed_video_num,
                    'max_mem': max_mem,
                    "max_mem_mb":max_mem_mb
                })
            print('Finished the evaluation on GPU {}.'.format(self.gpu))
            if self.rank == 0:
                for _ in range(self.gpu_num - 1):
                    info_dict = self.info_queue.get()
                    total_time += info_dict['total_time']
                    total_frame += info_dict['total_frame']
                    total_sfps += info_dict['total_sfps']
                    processed_video_num += info_dict['processed_video_num']
                    max_mem = max(max_mem, info_dict['max_mem'])
                    max_mem_mb = max(max_mem_mb, info_dict['max_mem_mb'])
                all_reduced_total_avg_time_per_frame = total_time / total_frame
                all_reduced_avg_sfps = total_sfps / processed_video_num
                print(
                    "GPU {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                    .format(list(range(self.gpu_num)),
                            1. / all_reduced_total_avg_time_per_frame,
                            1. / all_reduced_avg_sfps, max_mem))
                record_txt = cfg.DIR_EVALUATION + "/" + cfg.TEST_DATASET + "/inference_info_record.txt"
                with open(record_txt, mode='a+') as f:
                    f.write("\n------------------------\n")
                    f.write("All-Frame FPS: " + str(1. / all_reduced_total_avg_time_per_frame) + "\n")
                    f.write("All-Seq FPS: " + str(1. / all_reduced_avg_sfps) + "\n")
                    f.write("Max Mem: " + str(max_mem) + "\n")
                    f.write("Max Mem mb: " + str(max_mem_mb) + "\n")
        else:
            print(
                "GPU {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                .format(self.gpu, 1. / total_avg_time_per_frame, 1. / avg_sfps,
                        max_mem))
            print(
                "GPU {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                .format(self.gpu, 1. / total_avg_time_per_frame, 1. / avg_sfps,
                        max_mem))
            record_txt = cfg.DIR_EVALUATION + "/" + cfg.TEST_DATASET + "/inference_info_record.txt"
            with open(record_txt, mode='a+') as f:
                f.write("\n------------------------\n")
                f.write("All-Frame FPS: " + str(1. / total_avg_time_per_frame) + "\n")
                f.write("All-Seq FPS: " + str(1. / avg_sfps) + "\n")
                f.write("Max Mem: " + str(max_mem) + "\n")
                f.write("Max Mem mb: " + str(max_mem_mb) + "\n")

        if self.rank == 0 and (cfg.TEST_DATASET == 'mose' or cfg.TEST_DATASET == 'davis2017'):
            zip_folder_mose(self.source_folder, self.zip_dir)
            self.print_log('Saving result to {}.'.format(self.zip_dir))
            if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                zip_folder(self.result_root_sparse, self.zip_dir_sparse)
        elif self.rank == 0:
            zip_folder(self.source_folder, self.zip_dir)
            self.print_log('Saving result to {}.'.format(self.zip_dir))
            if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                zip_folder(self.result_root_sparse, self.zip_dir_sparse)
            end_eval_time = time.time()
            total_eval_time = str(
                datetime.timedelta(seconds=int(end_eval_time -
                                               start_eval_time)))
            self.print_log("Total evaluation time: {}".format(total_eval_time))

    def print_log(self, string):
        if self.rank == 0:
            print(string)
