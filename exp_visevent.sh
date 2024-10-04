################### X-Prompt RGB-E training with VisEvent ####################
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_accum.py \
    --amp \
    --exp_name default \
    --stage peft_visevent \
    --model default_onevos \
    --gpu_num 4 \
    --batch_size 4

################### evaluation on VisEvent ####################
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/eval.py \
    --amp \
    --exp_name default \
    --stage pre_ytb_dav \
    --model default_onevos \
    --dataset visevent \
    --split val \
    --ckpt_path "outputs/train/default_OneVOS/PEFT_VISEVENT/ema_ckpt/save_step_20000.pth" \
    --gpu_num 4

python tools/benchmark.py \
    -g ../datasets/VisEvent_new/test/Annotations \
    -r outputs/eval/visevent/visevent_val_default_OneVOS_PRE_YTB_DAV_ckpt_unknown