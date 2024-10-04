import torch.nn as nn

from networks.decoders import build_decoder
#from networks.cores.convmae.convmae_selfandcross_mem_idv2_patchasbackbone import build_mix_convmae
#from networks.cores.convmae.convmae_selfandcross_mem_idv2_patchasbackbone_para import build_mix_convmae
#from networks.cores.convmae.convmae_mix_decoup_mem_idv2_patchasbackbone import build_mix_convmae
#from networks.cores.convmae.convmae_mix_9_1_1base_idadd_fromlayer4 import build_mix_convmae
from networks.cores.convmae.convmae_mix_decoup_mem_idv2_patchasbackbone_shortcutright import build_mix_convmae


class AOT_CONVMAE(nn.Module):
    def __init__(self, cfg, decoder='fpn_vitshortcut_deconv_convmae'):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.backbone=build_mix_convmae(cfg)
        #decoder_indim =cfg.BACKBONE.DIM_EMBED[2]
        decoder_indim=cfg.BACKBONE.EMBED_DIM[-1]

        self.decoder = build_decoder(
            decoder,
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            #decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.BACKBONE.EMBED_DIM[-1]//2,
            shortcut_dims=cfg.BACKBONE.EMBED_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS)


        self.patch_wise_id_bank = nn.Conv2d(
            cfg.MODEL_MAX_OBJ_NUM + 1,
            cfg.BACKBONE.EMBED_DIM[-1],
            kernel_size=16,
            stride=16,
            padding=0)

        self.id_dropout = nn.Dropout(cfg.TRAIN_ID_DROPOUT, True)

        self._init_weight()

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_dropout(id_emb)
        return id_emb,id_emb.shape[2]*id_emb.shape[3]


    def decode_id_logits(self, backbone_outputs):
        #n, c, h, w = backbone_outputs[-1].size()
        pred_logit = self.decoder([backbone_outputs[-1]], backbone_outputs)
        return pred_logit

    def backbone_forward(self,img, prev_patch, id_embs_prev,
                          mem_k=None,mem_v=None,is_train=True,is_first=False):
        if is_first:
            search_patch_record, _, _, _ = self.backbone(prev_patch, id_embs_prev, img, mem_k=mem_k,mem_v=mem_v,is_train=is_train)
            return search_patch_record
        else:
            search_patch_record, search_features, newk, newv = self.backbone(prev_patch, id_embs_prev, img, mem_k=mem_k,mem_v=mem_v,is_train=is_train)
            return search_patch_record, search_features, newk, newv

    def _init_weight(self):
        #nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(
                self.cfg.BACKBONE.EMBED_DIM[-1], -1).permute(0, 1),
            gain=16**-2)