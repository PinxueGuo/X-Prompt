################### X-Prompt RGB-T training with VisT300 and UT-UAV ####################
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_accum.py \
    --amp \
    --exp_name XPrompt-RGBT \
    --stage peft_rgbt \
    --model default_onevos \
    --gpu_num 4 \
    --batch_size 4

################### evaluation on VisT300 ####################
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/eval.py \
    --amp \
    --exp_name XPrompt-RGBT \
    --stage pre_ytb_dav \
    --model default_onevos \
    --dataset vist300 \
    --split val \
    --ckpt_path "outputs/train/XPrompt-RGBT_OneVOS/PEFT_RGBT/ema_ckpt/save_step_60000.pth" \
    --gpu_num 4

python tools/benchmark.py \
    -g ../datasets/VisT300/test/Annotations \
    -r outputs/eval/vist300/vist300_val_XPrompt-RGBT_OneVOS_PRE_YTB_DAV_ckpt_unknown

#################### evaluation on VT-UAV ####################
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/eval.py \
    --amp \
    --exp_name XPrompt-RGBT \
    --stage pre_ytb_dav \
    --model default_onevos \
    --dataset vtuva \
    --split val \
    --ckpt_path "outputs/train/XPrompt-RGBT_OneVOS/PEFT_RGBT/ema_ckpt/save_step_60000.pth" \
    --gpu_num 4

python tools/benchmark.py \
    -g ../datasets/VT-UVA/test/Annotations \
    -r outputs/eval/vtuva/vtuva_val_XPrompt-RGBT_OneVOS_PRE_YTB_DAV_ckpt_unknown
