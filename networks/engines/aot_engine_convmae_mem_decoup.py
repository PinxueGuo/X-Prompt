import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict,List

from utils.math import generate_permute_matrix
from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d

from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
from torchvision.utils import make_grid, save_image
import os


class AOTEngine_CONVMAE(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,seq_name=""):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None

        self.restart_engine()
        print("train aot engine CONVMAE vitshortcut mem ")
        self.seq_name = seq_name
        self.id_bank_vis=1

    def forward(self,
                ref_imgs,
                prev_imgs,
                curr_imgs,
                all_frames,
                all_masks,
                batch_size,
                obj_nums,
                step=0,
                tf_board=False,
                use_prev_pred=False,
                enable_prev_frame=False,
                use_prev_prob=False):  # only used for training

        if self.losses is None:
            self._init_losses()

        self.freeze_id = True if use_prev_pred else False

        self.offline_mask_process(all_masks)
        self.total_offline_frame_num=self.cfg.DATA_SEQ_LEN

        # generate id_embs for ref and prev
        #self.id_embs_ref=self.id_embedding(frame_step=0, img=ref_imgs,obj_nums=obj_nums)
        self.id_embs_prev=self.id_embedding(frame_step=0, img=ref_imgs,obj_nums=obj_nums)

        curr_losses, curr_masks = [], []
        # generate mask of prev frame
        self.match_propogate_one_frame(prev_imgs, ref_imgs)
        self.update_mem()

        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[0][self.frame_step], step, return_prob=True)
        self.update_prev_frame(
            curr_mask if not use_prev_prob else curr_prob,
            None if use_prev_pred else [self.assign_identity(
                self.offline_one_hot_masks[0][self.frame_step])])
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)

        self.match_propogate_one_frame(curr_imgs[0], prev_imgs)
        self.update_mem()

        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[0][self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)
        self.update_prev_frame(
            curr_mask if not use_prev_prob else curr_prob,
            None if use_prev_pred else [self.assign_identity(
                self.offline_one_hot_masks[0][self.frame_step])])

        #print("self.cfg.TRAIN_MEM_EVERY:",self.cfg.TRAIN_MEM_EVERY)
        for now in range(1,self.total_offline_frame_num - 2):
            self.match_propogate_one_frame(curr_imgs[now],curr_imgs[now-1])
            if ((now+1) % self.cfg.TRAIN_MEM_EVERY)==0:
                #print("enter ",now)
                self.update_mem()

            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[0][self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)
            self.update_prev_frame(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else [self.assign_identity(
                    self.offline_one_hot_masks[0][self.frame_step])])

        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)
        loss = pred_loss
        all_pred_mask = curr_masks
        all_frame_loss = curr_losses

        boards = {'image': {}, 'scalar': {}} # type:Dict[str,Dict[str,List]]

        return loss, all_pred_mask, all_frame_loss, boards

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(
            cfg.TRAIN_TOP_K_PERCENT_PIXELS,
            cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]

        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        #self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        #self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5

    def update_mem(self):
        if self.memk is None:
            self.memk=self.new_memk
            self.memv=self.new_memv
        else:
            for i in range(len(self.memk)):
                self.memk[i] = torch.cat((self.memk[i], self.new_memk[i]), dim=2)
                self.memv[i] = torch.cat((self.memv[i], self.new_memv[i]), dim=2)
            #self.new_memv=None
            #self.new_memk=None

    def update_mem_eval(self):
        if self.memk is None:
            self.memk=self.new_memk
            self.memv=self.new_memv
        else:
            newk=self.new_memk
            newv=self.new_memv
            for i in range(len(self.memk)):
                N=newk[i].shape[2]
                Nf=max((self.ref_num-1)*N,N)
                if self.memk[i].shape[2]>=self.cfg.mem_capacity*N:
                    self.memk[i] =torch.cat(
                        (self.memk[i][:,:,:Nf], self.memk[i][:,:,-((self.cfg.mem_capacity-1)*(N)):]), 
                        dim=2
                    )
                    self.memv[i] = torch.cat(
                        (self.memv[i][:,:,:Nf], self.memv[i][:,:,-((self.cfg.mem_capacity-1)*(N)):]), 
                        dim=2
                    )

                    if self.add_pre:
                        self.memk[i] = torch.cat((newk[i],self.memk[i]), dim=2)
                        self.memv[i] = torch.cat((newv[i],self.memv[i]), dim=2)
                    else:
                        self.memk[i] = torch.cat((self.memk[i], newk[i]), dim=2)
                        self.memv[i] = torch.cat((self.memv[i], newv[i]), dim=2)
                else:
                    if self.add_pre:
                        self.memk[i] = torch.cat((newk[i],self.memk[i]), dim=2)
                        self.memv[i] = torch.cat((newv[i],self.memv[i]), dim=2)
                    else:
                        self.memk[i]=torch.cat((self.memk[i],newk[i]),dim=2)
                        self.memv[i] = torch.cat((self.memv[i], newv[i]), dim=2)
            self.add_pre=False

    def add_ref_num(self):
        self.ref_num+=1
        self.add_pre=True

    def process_one_mask(self, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
        else:
            curr_one_hot_mask = None

        return curr_one_hot_mask

    def offline_mask_process(self, all_masks=None):
        if all_masks is not None:
            # extract mask embeddings
            # Num_of_frames,B,C,H,W
            offline_one_hot_masks = one_hot_mask(all_masks, self.max_obj_num)
            offline_masks = list(
                torch.split(all_masks, self.batch_size, dim=0))
            offline_one_hot_masks = list(
                torch.split(offline_one_hot_masks, self.batch_size, dim=0))

        self.offline_one_hot_masks = [offline_one_hot_masks]
        self.offline_masks = [offline_masks]


    def assign_identity(self, one_hot_mask):
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum('bohw,bot->bthw', one_hot_mask,
                                        self.id_shuffle_matrix)

        id_emb,enc_hw = self.AOT.get_id_emb(one_hot_mask)

        #self.show_feature_map(id_emb,note="id_bank")

        id_emb=id_emb.view(self.batch_size, -1,enc_hw ).permute(2, 0, 1)

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def multi_id_embedding(self,
                            img=None,
                            mask=None,
                            frame_step=-1,
                            obj_nums=None,
                            img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step


        curr_id_embs=[]
        for i in range(3):
            curr_id_emb = self.assign_identity(self.offline_one_hot_masks[i][frame_step])
            curr_id_embs.append(curr_id_emb)

        return curr_id_embs

    def id_embedding(self,
                            img=None,
                            mask=None,
                            frame_step=-1,
                            obj_nums=None,
                            img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step


        curr_id_embs=[]
        # stage 1 id embedding
        curr_id_emb = self.assign_identity(self.offline_one_hot_masks[0][frame_step])
        curr_id_embs.append(curr_id_emb)

        return curr_id_embs


    def update_long_term_memory(self, new_long_term_memories):
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory,
                                     last_long_term_memory):
                updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def update_prev_frame(self,mask, id_emb=None):
        if id_emb is None:
            if len(mask.size()) == 3 or mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
            else:
                curr_one_hot_mask = mask
            id_emb = self.assign_identity(curr_one_hot_mask)
            self.id_embs_prev = [id_emb]
        else:
            self.id_embs_prev = id_emb

        #self.show_feature_map(self.id_embs_prev)

    def process_reference_frame(self,img, mask,obj_nums):
        self.obj_nums=obj_nums
        self.input_size_2d=img.size()[2:]

        #self.curr_one_hot_mask = self.process_one_mask(mask)
        #self.id_embs_ref = self.assign_identity(self.curr_one_hot_mask)
        self.id_embs_prev =  self.assign_identity(self.process_one_mask(mask))


    def show_feature_map(self,feature_map,note):
        #self.seq_name = self.seq_name
        save_root=self.cfg.DIR_EVALUATION+"/vis_davis2017/"+self.seq_name+"/"
        image_path=save_root+"/"+note
        #i2=save_root+"/"+note+"/0"
        #i3 = save_root + "/" + note + "/1"

        if note == "id_bank":
            for path in [
                image_path
            ]:
                if not os.path.isdir(path):
                    try:
                        os.makedirs(path)
                    except Exception as inst:
                        print(inst)
                        print('Failed to make dir: {}.'.format(path))
            final_path = image_path + "/" + str(self.id_bank_vis)  + ".png"
            #save_image(feature_map.transpose(0, 1), final_path, normalize=False, nrow=16)
            save_image(feature_map[:,0:256,].transpose(0, 1), final_path, normalize=False, nrow=16)
            self.id_bank_vis+=1


        if note=="shortcut":
            for path in [
                image_path
            ]:
                if not os.path.isdir(path):
                    try:
                        os.makedirs(path)
                    except Exception as inst:
                        print(inst)
                        print('Failed to make dir: {}.'.format(path))

        if isinstance(feature_map,list):
            if note=="first_6_stage":
                j = 0
                for i in range(0,len(feature_map) ,1):
                    now = save_root + "/" + note + "/"+str(i)
                    if not os.path.isdir(now):
                        try:
                            os.makedirs(now)
                        except Exception as inst:
                            print(inst)
                            print('Failed to make dir: {}.'.format(now))
                    final_path = image_path + "/" + str(i) + "/" + str(self.vis_first6stage[j]) + ".png"
                    save_image(feature_map[i][:,0:256,].transpose(0, 1), final_path, normalize=False, nrow=16)
                    self.vis_first6stage[j] += 1
                    j+=1
            elif note=="last_6_stage":
                j = 0
                for i in range(0, len(feature_map) , 1):
                    now = save_root + "/" + note + "/" + str(i)
                    if not os.path.isdir(now):
                        try:
                            os.makedirs(now)
                        except Exception as inst:
                            print(inst)
                            print('Failed to make dir: {}.'.format(now))
                    final_path = image_path + "/" + str(i) + "/" + str(self.vis_last6stage[j]) + ".png"
                    save_image(feature_map[i][:,0:256,].transpose(0, 1), final_path, normalize=False, nrow=16)
                    self.vis_last6stage[j] += 1
                    j += 1

    def match_propogate_one_frame(self, img=None, prev_image=None, is_train=True):
        self.frame_step += 1
        if self.frame_step==1:
            # only for patching embedding, without network. this output patch_record is the prev_frame
            self.patch_record = self.AOT.backbone_forward(img,
                                                          prev_image,
                                                          self.id_embs_prev,
                                                          mem_k=self.memk,
                                                          mem_v=self.memv,
                                                          is_train=is_train,
                                                          is_first=True)
            # run network. this output patch_record is the current_frame
            self.patch_record, self.curr_search_features, self.new_memk, self.new_memv = self.AOT.backbone_forward(img, 
                                                                                                                   self.patch_record,
                                                                                                                   self.id_embs_prev,
                                                                                                                   mem_k=self.memk,
                                                                                                                   mem_v=self.memv,
                                                                                                                   is_train=is_train)
        else:
            #print("frame_steg:",self.frame_step)
            self.patch_record, self.curr_search_features, self.new_memk, self.new_memv = self.AOT.backbone_forward(img,
                                                                                                                   self.patch_record,
                                                                                                                   self.id_embs_prev,
                                                                                                                   mem_k=self.memk,
                                                                                                                   mem_v=self.memv,
                                                                                                                   is_train=is_train)

        #self.show_feature_map(self.curr_backbone_output)


    def decode_current_logits(self, output_size=None):
        #curr_backbone_outputs = self.curr_backbone_output_record
        #curr_backbone_outputs = self.curr_backbone_output

        pred_id_logits = self.AOT.decode_id_logits(self.curr_search_features)

        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum('bohw,bto->bthw', pred_id_logits,
                                          self.id_shuffle_matrix)

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):
            pred_id_logits[batch_idx, (obj_num+1):] = - \
                1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4

        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(pred_id_logits,
                                           size=output_size,
                                           mode="bilinear",
                                           align_corners=self.align_corners)

        return pred_id_logits

    def predict_current_mask(self, gt_mask,output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(self.pred_id_logits,
                                       size=gt_mask.shape[2],
                                       mode="bilinear",
                                       align_corners=self.align_corners)
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def calculate_current_loss(self, gt_mask, step):
        pred_id_logits = self.pred_id_logits

        pred_id_logits = F.interpolate(pred_id_logits,
                                       size=gt_mask.size()[-2:],
                                       mode="bilinear",
                                       align_corners=self.align_corners)

        label_list = []
        logit_list = []
        for batch_idx, obj_num in enumerate(self.obj_nums):
            now_label = gt_mask[batch_idx].long()
            now_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
            label_list.append(now_label.long())
            logit_list.append(now_logit)

        total_loss = 0
        for loss, loss_weight in zip(self.losses, self.loss_weights):
            total_loss = total_loss + loss_weight * \
                loss(logit_list, label_list, step)

        return total_loss

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        self.decode_current_logits()
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            mask, prob = self.predict_current_mask(gt_mask,return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask(gt_mask)
            return loss, mask

    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(
            shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        self.memk=None
        self.memv=None
        self.ref_num=1
        self.add_pre=False

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(
                self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None


    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class AOTInferEngine_CONVMAE(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None,seq_name=""):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip

        self.aot_engines = []
        self.seq_name = seq_name

        self.restart_engine()

    def restart_engine(self):
        self.aot_engines = []
        self.obj_nums = None


    def separate_mask(self, mask):
        if mask is None:
            return [None] * len(self.aot_engines)
        if len(self.aot_engines) == 1:
            return [mask]

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs

    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1:1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(torch.cat(bg_logits, dim=1),
                                dim=1,
                                keepdim=True)
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)

        return merged_logit

    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_aot_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob] + fg_probs,
                                dim=1).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def process_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]

        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = AOTEngine_CONVMAE(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip,self.seq_name)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks = self.separate_mask(mask)
        img_embs = None
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.process_reference_frame(img, separated_mask, obj_nums=[self.max_aot_obj_num])

        self.input_size_2d = self.aot_engines[0].input_size_2d


    def match_propogate_one_frame(self, img=None, prev_img=None, ):
        img_embs = None
        for aot_engine in self.aot_engines:
            aot_engine.match_propogate_one_frame(img, prev_img, is_train=False)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for aot_engine in self.aot_engines:
            all_logits.append(aot_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_prev_frame(self, curr_mask):
        separated_masks = self.separate_mask(curr_mask)
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.update_prev_frame(separated_mask)

    def update_mem(self):
        #separated_masks = self.separate_mask(curr_mask)
        for aot_engine in self.aot_engines:
            #aot_engine.update_mem(aot_engine.new_memk, aot_engine.new_memv)
            #aot_engine.update_mem()
            aot_engine.update_mem_eval()

    def add_ref_num(self):
        #separated_masks = self.separate_mask(curr_mask)
        for aot_engine in self.aot_engines:
            #aot_engine.update_mem(aot_engine.new_memk, aot_engine.new_memv)
            #aot_engine.update_mem()
            aot_engine.add_ref_num()

