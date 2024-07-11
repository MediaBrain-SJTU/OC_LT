# T2H
This is the official code base for "Long-Tailed Diffusion Models With Oriented Calibration" accepted as ICLR2024 poster.
## Abstract
Diffusion models are acclaimed for generating high-quality and diverse images. However, their performance notably degrades when trained on data with a long-tailed distribution. For long tail diffusion model generation, current works focus on the calibration and enhancement of the tail generation with head-tail knowledge transfer. The transfer process relies on the abundant diversity derived from the head class and, more significantly, the condition capacity of the model prediction. However, the dependency on the conditional model prediction to realize the knowledge transfer might exhibit bias during training, leading to unsatisfactory generation results and lack of robustness. Utilizing a Bayesian framework, we develop a weighted denoising score-matching technique for knowledge transfer directly from head to tail classes. Additionally, we incorporate a gating mechanism in the knowledge transfer process. We provide statistical analysis to validate this methodology, revealing that the effectiveness of such knowledge transfer depends on both label distribution and sample similarity, providing the insight to consider sample similarity when re-balancing the label proportion in training. We extensively evaluate our approach with experiments on multiple benchmark datasets, demonstrating its effectiveness and superior performance compared to existing methods.

## Attention
This repository is being organized and updated continuously. Please note that this version is not the final release.

## Running
This code base heavily depend on CBDM(https://github.com/qym7/CBDM-pytorch)
### Training
```
CUDA_VISIBLE_DEVICES=1 python main.py --train --transfer_x0  --transfer_mode t2h  --data_type cifar10lt --num_class 10 --logdir ./logs --cfg --conditional
```
### Evaluation
first put the /stats folder from CBDM code base under our folder then run

```
CUDA_VISIBLE_DEVICES=1 python ddpm_gen.py --eval --ckpt_step xxx --w 1.5 --conditional --cfg --num_class 10 --logdir ./logs
```
ckpt step is the checkpoint saving time, we use 200000 for default setting.

### Cite 
```
@inproceedings{zhang2024long,
  title={Long-tailed diffusion models with oriented calibration},
  author={Zhang, Tianjiao and Zheng, Huangjie and Yao, Jiangchao and Wang, Xiangfeng and Zhou, Mingyuan and Zhang, Ya and Wang, Yanfeng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
