################### X-Prompt RGB-D training with ARKitTrack ####################
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_accum.py \
    --amp \
    --exp_name XPrompt-RGBD \
    --stage peft_arkittrack \
    --model default_onevos \
    --gpu_num 4 \
    --batch_size 4

################### evaluation on ARKitTrack ####################
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/eval.py \
    --amp \
    --exp_name XPrompt-RGBD \
    --stage pre_ytb_dav \
    --model default_onevos \
    --dataset arkittrack \
    --split val \
    --ckpt_path "outputs/train/XPrompt-RGBD_OneVOS/PEFT_ARKITTRACK/ema_ckpt/save_step_24000.pth" \
    --gpu_num 4 \
    --expert_num 3

python tools/benchmark.py \
    -g ../datasets/ARKitTrack/test/Annotations \
    -r outputs/eval/arkittrack/arkittrack_val_XPrompt-RGBD_OneVOS_PRE_YTB_DAV_ckpt_unknown