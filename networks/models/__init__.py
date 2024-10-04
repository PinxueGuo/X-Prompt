
# from networks.models.aot_convmae_andmem_decoup_eval import AOT_CONVMAE_eval
from networks.models.aot_convmae_andmem_decoup import AOT_CONVMAE


def build_vos_model(name, cfg, **kwargs):
    if name == 'aot_convmae':
        return AOT_CONVMAE(cfg, **kwargs)
    # if name == 'aot_convmae_eval':
    #     return AOT_CONVMAE_eval(cfg, **kwargs)
    else:
        raise NotImplementedError