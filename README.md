# X-Prompt: Multi-modal Visual Prompt for Video Object Segmentation
This repository contains the code for the paper X-Prompt: Multi-modal Visual Prompt for Video Object Segmentation [ACMMM'2024].

## Install
```bash
conda create -n vos python=3.9 -y
conda activate vos
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -
```

## Prepare
### Dataset
Reformed and preprocessed datasets that can be used directly:
- [RGB-T] [VisT300](https://pan.baidu.com/s/1Ya25RGHmT_ZbzbMB-HE3fQ?pwd=uqj7)
- [RGB-T] [VT-UVA](https://pan.baidu.com/s/1UVsN_WR9lixZYsUpiOkekA?pwd=3mm6)
- [RGB-D] [ARKitTrack](https://pan.baidu.com/s/1i6ctcSXhkzHxx0s8kLFrmw?pwd=9h6g)
- [RGB-E] Since the VisEvent dataset only provides bounding box annotations and lacks mask annotations, we initially used HQ-SAM to generate masks for our experiments. Now we notice a new RGB-E dataset [LLE-VOS](https://github.com/HebeiFast/EventLowLightVOS) annoted with masks for VOS has been released. We recommend using LLE-VOS directly, and we will be updating our work with experiments on the RGB-E data from LLE-VOS.

```bash
├── X-Prompt
├── datasets
│   ├── VisT300
│   │    └── ...
│   ├── VT-UVA
│   │    └── ...
│   ├── ARKitTrack
│   │    └── ...
│   ├── ...
```

### Pretrained Foudantion Model
- Pretrained RGB VOS Foundation [Model](https://pan.baidu.com/s/1tAPghe_CXuUM_r02olNIeg?pwd=scw4)

Place the weights in X-Prompt/weights. 
We pretrain the RGB foundation model following the OneVOS [ECCV'2024]. More details on the model design and training, please refer to [OneVOS](https://github.com/L599wy/OneVOS).

## Train & Eval
```bash
# [RGB-T] VisT300 and VT-UAV
bash exp_rgbt.sh

# [RGB-D] ARKitTrack
bash exp_arkittrack.sh
```

We also provide our trained models you can use directly for inference and evaluation. 
- [X-Prompt RGB-T](https://pan.baidu.com/s/1H1bYkSQoHkoueDJNj1cUlQ?pwd=kqgh)
- [X-Prompt RGB-D](https://pan.baidu.com/s/1dKgHqLL6wzlIbEJJKONMrg?pwd=tiku)

Note that we didn't thoroughly search for hyperparameters, so there may be better choices to get better performance than reported.

## Citations
If you find this repository useful, please consider giving a star and citation:
```bibtex
@inproceedings{guo2024x,
  title={X-Prompt: Multi-modal Visual Prompt for Video Object Segmentation},
  author={Guo, Pinxue and Li, Wanyun and Huang, Hao and Hong, Lingyi and Zhou, Xinyu and Chen, Zhaoyu and Li, Jinglun and Jiang, Kaixun and Zhang, Wei and Zhang, Wenqiang},
  booktitle={ACM Multimedia 2024}
}

@article{li2024onevos,
  title={OneVOS: Unifying Video Object Segmentation with All-in-One Transformer Framework},
  author={Li, Wanyun and Guo, Pinxue and Zhou, Xinyu and Hong, Lingyi and He, Yangji and Zheng, Xiangyu and Zhang, Wei and Zhang, Wenqiang},
  journal={arXiv preprint arXiv:2403.08682},
  year={2024}
}
```
