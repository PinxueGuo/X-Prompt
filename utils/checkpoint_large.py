import math

import numpy
import torch
import os
import numpy as np
import torch.nn.functional as F


def interpolate_pos_encoding(pos_embed, h, w):
    #npatch = x.shape[1]
    N = pos_embed.shape[1]
    if N == h*w and w == h:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    # w0 = w // self.patch_size
    # h0 = h // self.patch_size
    w0 = w
    h0 = h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


def load_network_and_optimizer(net, opt, pretrained_dir, gpu, scaler=None):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    opt.load_state_dict(pretrained['optimizer'])
    if scaler is not None and 'scaler' in pretrained.keys():
        scaler.load_state_dict(pretrained['scaler'])
    del (pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove


def load_network_and_optimizer_v2(net, opt, pretrained_dir, gpu, scaler=None):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    # load model
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)

    # load optimizer
    opt_dict = opt.state_dict()
    all_params = {
        param_group['name']: param_group['params'][0]
        for param_group in opt_dict['param_groups']
    }
    pretrained_opt_dict = {'state': {}, 'param_groups': []}
    for idx in range(len(pretrained['optimizer']['param_groups'])):
        param_group = pretrained['optimizer']['param_groups'][idx]
        if param_group['name'] in all_params.keys():
            pretrained_opt_dict['state'][all_params[
                param_group['name']]] = pretrained['optimizer']['state'][
                    param_group['params'][0]]
            param_group['params'][0] = all_params[param_group['name']]
            pretrained_opt_dict['param_groups'].append(param_group)

    opt_dict.update(pretrained_opt_dict)
    opt.load_state_dict(opt_dict)

    # load scaler
    if scaler is not None and 'scaler' in pretrained.keys():
        scaler.load_state_dict(pretrained['scaler'])
    del (pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove


def load_network(net, pretrained_dir, gpu):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del (pretrained)
    return net.cuda(gpu), pretrained_dict_remove

def load_by_npz(net, pretrained_dir, gpu):
    pretrained=np.load(pretrained_dir)
    pretrained_dict=pretrained.files
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    model_dict = net.backbone.state_dict()

    for k in pretrained_dict:
        if k in model_dict:
            print("havs:",k)
            pretrained_dict_update[k] = pretrained_dict[k]
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = pretrained_dict[k]
        elif k[:9] == 'backbone.':
            if k[9:] in model_dict:
                pretrained_dict_update[k[9:]] = pretrained_dict[k]
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.backbone.load_state_dict(model_dict)
    del (pretrained)
    return net.cuda(gpu), pretrained_dict_remove

def load_backbone(net, pretrained_dir, gpu):
    if pretrained_dir[-4:]==".npz":
        load_by_npz(net, pretrained_dir, gpu)
    else:
        pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained
    model_dict = net.backbone.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            #print("havs:",k[0:13])
            pretrained_dict_update[k] = v
            if k=="pos_embed":
                print("pos_embed")
                new_v=v[:,1:,:]
                need_now = 24
                #need_now = 33
                pretrained_dict_update[k]=interpolate_pos_encoding(new_v,need_now,need_now)
            if "attn.kv" in k:
                Index = k.find("attn")
                dim = v.shape[0]
                pretrained_dict_update[k[:Index] + "attn.k" + k[Index + 7:]] = v[0:dim // 2]
                pretrained_dict_update[k[:Index] + "attn.v" + k[Index + 7:]] = v[dim // 2:]
            '''if k[0:13] == "blocks.0.attn":
                pretrained_dict_update["blocks.0.pre_id_attn" + k[13:]] = v'''
            '''if "qkv.weight" in k and ("1" in k or "3" in k or "5" in k or "7" in k or "9" in k or "11" in k) and "10" not in k:
                Index = k.find("qkv")
                pretrained_dict_update[k[:Index]+"linear_QK"+k[Index+3:]]=v[0:768*2,:]
                pretrained_dict_update[k[:Index] + "linear_V" + k[Index + 3:]] = v[768 * 2:, :]
                pretrained_dict_update[k[:Index] + "linear_Vid" + k[Index + 3:]] = v[768 * 2:, :]
            if "qkv.bias" in k and ("1" in k or "3" in k or "5" in k or "7" in k or "9" in k or "11" in k) and "10" not in k:
                Index = k.find("qkv")
                pretrained_dict_update[k[:Index] + "linear_QK" + k[Index + 3:]] = v[0:768 * 2]
                pretrained_dict_update[k[:Index] + "linear_V" + k[Index + 3:]] = v[768 * 2:]
                pretrained_dict_update[k[:Index] + "linear_Vid" + k[Index + 3:]] = v[768 * 2:]'''
            '''if "kv.weight" in k:
                Index = k.find("kv")
                s=v.shape[0]
                pretrained_dict_update[k[:Index] + "k" + k[Index + 2:]] = v[0:s//2, :]
                pretrained_dict_update[k[:Index] + "v" + k[Index + 3:]] = v[s//2:, :]'''
                #pretrained_dict_update[k[:Index] + "linear_Vid" + k[Index + 3:]] = v[768 * 2:, :]

            if k=="absolute_pos_embed":
                print("absolute_pos_embed")
                new_v = v
                need_now = 29
                # need_now = 33
                pretrained_dict_update[k] = interpolate_pos_encoding(new_v, need_now, need_now)
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        elif k[:9] == 'backbone.':
            if k[9:] in model_dict:
                pretrained_dict_update[k[9:]] = v
        elif "attn.kv" in k:
            Index = k.find("attn")
            dim = v.shape[0]
            pretrained_dict_update[k[:Index] + "attn.k" + k[Index + 7:]] = v[0:dim // 2]
            pretrained_dict_update[k[:Index] + "attn.v" + k[Index + 7:]] = v[dim // 2:]
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.backbone.load_state_dict(model_dict)
    differ = set(model_dict.keys()) ^ set(pretrained_dict.keys())
    print("diff keys:",differ)
    del (pretrained)
    return net.cuda(gpu), pretrained_dict_remove

def save_network(net,
                 opt,
                 step,
                 save_path,
                 max_keep=8,
                 backup_dir='./saved_models',
                 scaler=None):
    ckpt = {'state_dict': net.state_dict(), 'optimizer': opt.state_dict()}
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save(ckpt, save_dir)
    except:
        save_path = backup_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save(ckpt, save_dir)

    all_ckpt = os.listdir(save_path)
    if len(all_ckpt) > max_keep:
        all_step = []
        for ckpt_name in all_ckpt:
            step = int(ckpt_name.split('_')[-1].split('.')[0])
            all_step.append(step)
        all_step = list(np.sort(all_step))[:-max_keep]
        for step in all_step:
            ckpt_path = os.path.join(save_path, 'save_step_%s.pth' % (step))
            os.system('rm {}'.format(ckpt_path))
