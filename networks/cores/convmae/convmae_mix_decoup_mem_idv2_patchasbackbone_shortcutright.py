
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_
from itertools import repeat
import collections.abc

from lib.utils.misc import is_main_process
#from lib.models.mixformer_cvt.utils import to_2tuple
#from lib.models.mixformer_vit.pos_utils import get_2d_sincos_pos_embed
import math

from einops import rearrange
import loralib as lora
from loralib.layers import LoRALayer

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinearExperts(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_num: int = 1,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.lora_num = lora_num
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)
            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5)) 
            
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            route_weight = nn.functional.softmax(self.lora_route(x), dim=-1, dtype=torch.float32).to(result.dtype)
            for i in range(self.lora_num):
                result = result + torch.unsqueeze(route_weight[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,cfg=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = lora.MergedLinear(dim, dim * 3, r=cfg.EXPERT_LOW_RANK, enable_lora=[True, True, True])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearExperts(dim, dim, r=cfg.EXPERT_LOW_RANK, lora_num=cfg.EXPERT_NUM)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.linear_ID_KV = nn.Linear(dim, dim + self.num_heads)
        # self._init_weights(self.linear_ID_KV)
        # self.linear_ID_K = lora.Linear(dim, self.num_heads, r=cfg.EXPERT_LOW_RANK)
        # self.linear_ID_V = lora.Linear(dim, dim, r=cfg.EXPERT_LOW_RANK)
        self.linear_ID_K = LinearExperts(dim, self.num_heads, r=cfg.EXPERT_LOW_RANK, lora_num=1)
        self.linear_ID_V = LinearExperts(dim, dim, r=cfg.EXPERT_LOW_RANK, lora_num=1)

        self.dim=dim

        self.topk = cfg.is_topk
        print("istopk:", self.topk)
        if self.topk is True:
            self.topk_template = cfg.topk_tempalte
            self.topk_search = cfg.topk_search
            print("tokp_template:", self.topk_template)
            print("tokp_search:", self.topk_search)
            self.topk_percent = cfg.is_topk_percent
            print("istopkpercent:", self.topk)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def softmax_w_top(self, x, top):
        top=int(top)
        values, indices = torch.topk(x, k=top, dim=-1)
        x_exp = torch.exp(values[:,:,:,:] - values[:, :,:,0:1])
        x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(-1, indices, x_exp.type(x.dtype))  # B * THW * HW
        return x

    def key_value_id(self, id_emb, k_m, v_m):
        # ID_KV = self.linear_ID_KV(id_emb)
        # ID_K, ID_V = torch.split(ID_KV, [self.num_heads, self.dim], dim=2)
        ID_K = self.linear_ID_K(id_emb)
        ID_V = self.linear_ID_V(id_emb)
        k_m = k_m * ((1 + torch.tanh(ID_K)).transpose(1, 2).unsqueeze(-1))
        v_m = v_m.permute(0, 2, 1, 3).flatten(2) + ID_V
        return k_m, v_m

    def forward(self, x, id_total , mem_k, mem_v , id_add=False):
        B, N, C = x.shape
        N_m = id_total.shape[1]
        N_s = N - id_total.shape[1]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        v_m, v_s = torch.split(v, [N_m, N_s], dim=2)
        k_m, k_s = torch.split(k, [N_m, N_s], dim=2)
        v_add_id=v
        if id_add:
            k_m, v_m=self.key_value_id(id_total, k_m, v_m)
            v_m = v_m.reshape(B, N_m, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_add_id = torch.cat([v_m, v_s], dim=2)

        q = q * self.scale
        # divided q
        q_m, q_s = torch.split(q, [N_m, N_s], dim=2)

        # template attention
        attn = (q_m @ k_m.transpose(-2, -1))
        if self.topk:
            if self.topk_percent:
                topk=int(self.topk_template/100*k_m.shape[2])
            else:
                topk=self.topk_template
            attn = self.softmax_w_top(attn, top=topk)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_m = (attn @ v_m).transpose(1, 2).reshape(B, N_m, C)

        # search attention
        if id_add and mem_k is not None:
            k=torch.cat((mem_k, k),dim=2)
            v_add_id=torch.cat((mem_v,v_add_id),dim=2)

        attn = (q_s @ k.transpose(-2, -1))
        if self.topk:
            if self.topk_percent:
                topk = int(self.topk_search/100 * k.shape[2])
            else:
                topk = self.topk_search
            attn = self.softmax_w_top(attn, top=topk)
        else:
            #print("no topk")
            attn = attn.softmax(dim=-1)
            #print("atten:",attn.shape)
        attn = self.attn_drop(attn)
        x_s = (attn @ v_add_id).transpose(1, 2).reshape(B, N_s, C)


        x = torch.cat([x_m, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if id_add:
            return x,k_m,v_m
        else:
            return x,None,None


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,cfg=cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, template, search, id_template, bs, mem_k, mem_v, id_add):
        B, N, C = search.shape
        BT = template.shape[0]
        t_N = N
        if BT != bs:
            template = template.view(bs, -1, C)
            t_N = int(BT / bs) * N

        x = torch.cat([template, search], dim=1)

        # new_memk=None
        x1, new_memk, new_memv = self.attn(self.norm1(x), id_template.transpose(0, 1), mem_k, mem_v, id_add=id_add)
        x = x + self.drop_path1(x1)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        template, search = torch.split(x, [t_N, N], dim=1)

        if template.shape[0] != BT:
            template = template.view(BT, N, C)

        return template, search, new_memk, new_memv



class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=464, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
                 depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], in_chans=3, num_classes=1000,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None,cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # RGB patch_embed and conv block
        self.patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])

        # X-prompt patch_embed and conv block
        self.prompt_patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.prompt_patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.prompt_patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.prompt_patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.prompt_pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.prompt_blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.prompt_blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])

        # prompter that fuse rgb-patch and x-patch
        self.prompt_fuser1 = nn.Linear(embed_dim[0]*2, embed_dim[0])
        self.prompt_fuser2 = nn.Linear(embed_dim[1]*2, embed_dim[1])
        self.prompt_fuser3 = nn.Linear(embed_dim[2]*2, embed_dim[2])
        self.prompt_ca = ChannelAttention(embed_dim[2]*2)
        self.prompt_sa = SpatialAttention()
        self.prompt_fuser = nn.Linear(embed_dim[2]*2, embed_dim[2])


        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer,cfg=cfg)
            for i in range(depth[2])])
        # self.norm = norm_layer(embed_dim[-1])

        self.apply(self._init_weights)

        self.grid_size = img_size // (patch_size[0] * patch_size[1] * patch_size[2])
        self.num_patches = self.grid_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim[2]))

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        # w0 = w // self.patch_size
        # h0 = h // self.patch_size
        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    '''def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
                                              cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))'''

    def forward(self, template_patch, id_template, search, is_train, mem_k=None, mem_v=None, bs=None):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        rgb_template_patch = template_patch[:, :3]
        prompt_template_patch = template_patch[:, 3:]
        rgb_search = search[:, :3]
        prompt_search = search[:, 3:]

        if template_patch.shape[2]==search.shape[2]:
            ### conv embeddings for x_t
            rgb_template_patch = self.patch_embed1(rgb_template_patch)  # [b,d=256,H/4,W/4]
            rgb_template_patch = self.pos_drop(rgb_template_patch)
            for blk in self.blocks1:
                rgb_template_patch = blk(rgb_template_patch)
            prompt_template_patch = self.prompt_patch_embed1(prompt_template_patch)
            prompt_template_patch = self.prompt_pos_drop(prompt_template_patch)
            for blk in self.prompt_blocks1:
                prompt_template_patch = blk(prompt_template_patch)
            # fuse_template_patch = self.prompt_fuser1(
            #     torch.cat([rgb_template_patch, prompt_template_patch], dim=1).permute(0, 2, 3, 1)   # [b,h,w,d]
            # )

            rgb_template_patch = self.patch_embed2(rgb_template_patch)  # [b,d=384,H/8,W/8]
            for blk in self.blocks2:
                rgb_template_patch= blk(rgb_template_patch)
            prompt_template_patch = self.prompt_patch_embed2(prompt_template_patch)
            for blk in self.prompt_blocks2:
                prompt_template_patch = blk(prompt_template_patch)
            # fuse_template_patch = self.prompt_fuser2(
            #     torch.cat([rgb_template_patch, prompt_template_patch], dim=1).permute(0, 2, 3, 1)
            # )

            rgb_template_patch = self.patch_embed3(rgb_template_patch)  # [b,d=768,H/16,W/16]
            prompt_template_patch = self.prompt_patch_embed3(prompt_template_patch)
            # fuse_template_patch = self.prompt_fuser3(
            #     torch.cat([rgb_template_patch, prompt_template_patch], dim=1).permute(0, 2, 3, 1)
            # )

            # B, C = rgb_template_patch.size(0), rgb_template_patch.size(-1)
            H_t = rgb_template_patch.shape[2]
            W_t = rgb_template_patch.shape[3]

            rgb_template_patch = rgb_template_patch.flatten(2).permute(0, 2, 1)  # BCHW --> BNC
            rgb_template_patch = self.patch_embed4(rgb_template_patch)  # [b,d=768,H/16,W/16]
            prompt_template_patch = prompt_template_patch.flatten(2).permute(0, 2, 1)  # BCHW --> BNC
            prompt_template_patch = self.prompt_patch_embed4(prompt_template_patch)

            rgbx_template_patch = torch.cat([rgb_template_patch, prompt_template_patch], dim=-1)
            rgbx_template_patch = rearrange(rgbx_template_patch, 'b (h w) c -> b c h w', h=H_t, w=W_t)
            channel_attn = self.prompt_ca(rgbx_template_patch)   # b c 1 1
            spatial_attn = self.prompt_sa(rgbx_template_patch)   # b 1 h w
            sc_attn = channel_attn * spatial_attn   # b c h w
            attned_rgbx_template_patch = rgbx_template_patch + (rgbx_template_patch * sc_attn)    # b c h w
            fuse_template_patch = self.prompt_fuser(
                attned_rgbx_template_patch.flatten(2).permute(0, 2, 1)
            )   # [b,HW,d]

            pos_embed = self.pos_embed
            if pos_embed.shape[1] != H_t * W_t:
                pos_embed = self.interpolate_pos_encoding(fuse_template_patch, H_t, W_t)
            template_patch = fuse_template_patch + pos_embed

            return template_patch, None, None, None


        search_features = []

        ### conv embeddings for x_s
        rgb_search = self.patch_embed1(rgb_search)
        rgb_search = self.pos_drop(rgb_search)
        for blk in self.blocks1:
            rgb_search = blk(rgb_search)
        prompt_search = self.prompt_patch_embed1(prompt_search)
        prompt_search = self.prompt_pos_drop(prompt_search)
        for blk in self.prompt_blocks1:
            prompt_search = blk(prompt_search)
        fuse_search = self.prompt_fuser1(
            torch.cat([rgb_search, prompt_search], dim=1).permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)   # [b,d,h,w]
        search_features.append(fuse_search)

        rgb_search = self.patch_embed2(rgb_search)
        for blk in self.blocks2:
            rgb_search = blk(rgb_search)
        prompt_search = self.prompt_patch_embed2(prompt_search)
        for blk in self.prompt_blocks2:
            prompt_search = blk(prompt_search)
        fuse_search = self.prompt_fuser2(
            torch.cat([rgb_search, prompt_search], dim=1).permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        search_features.append(fuse_search)

        rgb_search = self.patch_embed3(rgb_search)
        prompt_search = self.prompt_patch_embed3(prompt_search)
        fuse_search = self.prompt_fuser3(
            torch.cat([rgb_search, prompt_search], dim=1).permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        search_features.append(fuse_search)

        # B, C = search.size(0), search.size(-1)
        H_s,W_s = rgb_search.shape[2], rgb_search.shape[3]

        rgb_search = rgb_search.flatten(2).permute(0, 2, 1) #BCHW --> BNC
        rgb_search = self.patch_embed4(rgb_search)
        prompt_search = prompt_search.flatten(2).permute(0, 2, 1)  # BCHW --> BNC
        prompt_search = self.prompt_patch_embed4(prompt_search)

        rgbx_patch = torch.cat([rgb_search, prompt_search], dim=-1)
        rgbx_patch = rearrange(rgbx_patch, 'b (h w) c -> b c h w', h=H_s, w=W_s)
        channel_attn = self.prompt_ca(rgbx_patch)   # b c 1 1
        spatial_attn = self.prompt_sa(rgbx_patch)   # b 1 h w
        sc_attn = channel_attn * spatial_attn   # b c h w
        attned_rgbx_patch = rgbx_patch + (rgbx_patch * sc_attn)    # b c h w
        fuse_search = self.prompt_fuser(
            attned_rgbx_patch.flatten(2).permute(0, 2, 1)
        )   # [b,hw,c]

        pos_embed = self.pos_embed
        if pos_embed.shape[1] != H_s * W_s:
            pos_embed = self.interpolate_pos_encoding(fuse_search, H_s, W_s)
        search  = fuse_search  + pos_embed


        self.search_patch_record = search   # as template patch for the next frame (saving patch-embed time)

        self.search_patch = self.pos_drop(search)
        self.template_patch = self.pos_drop(template_patch)

        now_layer=0
        new_memks = []
        new_memkv = []
        for blk in self.blocks3:
            if mem_k is not None:
                memk = mem_k[now_layer]
                memv = mem_v[now_layer]
            else:
                memk = None
                memv = None
            self.template_patch, self.search_patch, new_memk, new_memv  = blk(self.template_patch,
                                                                            self.search_patch,
                                                                            id_template,
                                                                            bs=bs,
                                                                            mem_k=memk,
                                                                            mem_v=memv,
                                                                            id_add=True)
            new_memks.append(new_memk)
            new_memkv.append(new_memv)
            now_layer += 1

        search_features.append(self.search_patch.transpose(1, 2).reshape(bs, -1, int(H_s), int(W_s)))

        return self.search_patch_record, search_features, new_memks, new_memkv

class Seg_Convmae(nn.Module):
    def __init__(self, backbone):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, template_patch,id_template, search,mem_k=None,mem_v=None,is_train=True):
        # search: (b, c, h, w)
        bs=search.shape[0]
        if search.dim() == 5:
            search = torch.flatten(search, start_dim=0, end_dim=1)

        if isinstance(id_template,list):
            id_template=id_template[0]

        search_patch_record, search_features,new_memks,new_memkv = self.backbone(template_patch,id_template, search,is_train,mem_k=mem_k,mem_v=mem_v,bs=bs)
        return search_patch_record,  search_features,new_memks,new_memkv



def get_convmae_model(config, **kwargs):
    msvit_spec = config.BACKBONE
    img_size = config.DATA_RANDOMCROP[0]

    if msvit_spec.VIT_TYPE == 'convmae_base':
        convViT = ConvViT(in_chans=3,
            img_size=img_size, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
            depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),cfg=config)
    elif msvit_spec.VIT_TYPE == 'convmae_large':
        convViT = ConvViT(in_chans=3,
            img_size=img_size, patch_size=[4, 2, 2], embed_dim=[384, 768, 1024],
            depth=[2, 2, 20], num_heads=16, mlp_ratio=[4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),cfg=config)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'convmae_base' or 'convmae_large'")

    return convViT




def build_mix_convmae(cfg):
    print("build mixattention convmae mem decoup idv2 shortcutright")
    backbone = get_convmae_model(cfg)  # backbone without positional encoding and attention mask
    #box_head = build_box_head(cfg)  # a simple corner head
    model = Seg_Convmae(
        backbone,
    )

    return model



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)