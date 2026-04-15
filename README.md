# CBDM_TS: Mitigating Exposure Bias in Long-tailed Diffusion Models through Improved Time-steps and Mean Estimation

This repo contains the PyTorch implementation for [Class-Balancing Diffusion Models], by Yiming Qin, Huangjie Zheng, Jiangchao Yao, Mingyuan Zhou, and Ya Zhang.

In the field of imbalanced image generation, long-tailed diffusion models have achieved outstanding results. However, there has been little research on the exposure bias problem that exists in long-tailed diffusion models. 
In this paper, we conduct a systematic study of the exposure bias in long-tailed diffusion models and find that this bias arises from the inconsistency in network inputs between the training and sampling phases. To mitigate this bias, we propose a method that requires no additional training and improves performance from two aspects: time-step selection and mean estimation. Regarding time-step selection, our study reveals that during sampling, for a given time-step $t$ and its corresponding $\hat{x}_t$, there may exist an alternative time-step $t_k$ that exhibits a stronger coupling with $\hat{x}_t$. Based on this observation, we adjust the sampling time-steps accordingly. Regarding mean estimation, we observe that in the sampling process of long-tailed diffusion models, there is gradient information between the estimated values of the original sample at two consecutive time-steps. By leveraging this gradient information, we extrapolate the two estimates to compute a more accurate overall estimate. Our method can be applied to various long-tailed diffusion models (e.g., CBDM, T2H) and is compatible with multiple samplers, including DDPM, DDIM, and other higher-order samplers. For instance, on the CIFAR100LT long-tailed dataset, our method achieves an impressive FID of 5.73 when using the DEIS sampler.

## About this repository
The repo is implemented based on (https://github.com/qym7/CBDM-pytorch). Currently it supports the training for four datasets namely CIFAR10(LT) and CIFAR100(LT) under following three mechanisms:

1. Regular (conditional or unconditional) diffusion model training
2. Class-balancing model training
3. Class-balancing model finetuning based on a regular diffusion model

## Running the Experiments
We provide mainly the scripts for trianing and evaluating the CIFAR100LT dataset.
To run the code, please change the argument 'root' to the path where the dataset is downloaded.

## Files used in evaluation

Please find the [features for cifar 100 and cifar 10](https://drive.google.com/drive/folders/1Y89vu9DGiQsHl8YvwMrr_7UT4p4Pg_wV?usp=sharing) used in precision/recall/f_beta metrics. Put them in the stats folder and the codes will be ready to run. Note that those two metrics will only be evaluated if the number of samples is 50k otherwise it returns 0.


### Train a model
* Class-balancing model training 
    ```
    python main.py --train  --flagfile ./config/cifar100.txt --parallel --logdir ./logs/cifar100lt_cbdm --total_steps 300001 --conditional --data_type cifar100lt --imb_factor 0.01 --img_size 32  --batch_size 48 --save_step 100000 --sample_step 50000 --cb --tau 1.0
    ```


### Evaluate a model
* Sample images and evaluate for the 4 models above.

    ```
    python main.py --flagfile ./logs/cifar100lt_cbdm/flagfile.txt --logdir ./logs/cifar100lt_cbdm --fid_cache ./stats/cifar100.train.npz --ckpt_step 300000 --num_images 50000 --batch_size 64 --notrain --eval --sample_method cfg  --omega 1.6
    ```


## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{qin2023class,
  title={Class-balancing diffusion models},
  author={Qin, Yiming and Zheng, Huangjie and Yao, Jiangchao and Zhou, Mingyuan and Zhang, Ya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Acknowledgements

This implementation is based on / inspired by:

- [https://github.com/w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) 
- [https://github.com/crowsonkb/k-diffusion/blob/master/train.py](https://github.com/crowsonkb/k-diffusion/blob/master/train.py) (we refer to the implementation of ADA augmentation in K-diffusion model).
