from networks.engines.aot_engine_convmae_mem_decoup import AOTInferEngine_CONVMAE, AOTEngine_CONVMAE

# # small object expand
# from networks.engines.aot_engine_convmae_mem_decoup_small_obj_new import AOTInferEngine_CONVMAE_eval, AOTEngine_CONVMAE_eval
# from networks.engines.aot_engine_convmae_mem_decoup_small_obj_new_droplastall import AOTInferEngine_CONVMAE_eval, AOTEngine_CONVMAE_eval

def build_engine(name, phase='train', **kwargs):
    if name=='aotengine_convmae':
        if phase == 'train':
            return AOTEngine_CONVMAE(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine_CONVMAE(**kwargs)
        else:
            raise NotImplementedError
            
    if name=='aotengine_convmae_small_eval':
        if phase == 'train':
            return AOTEngine_CONVMAE_eval(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine_CONVMAE_eval(**kwargs)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError