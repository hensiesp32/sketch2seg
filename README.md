https://github.com/lllyasviel/ControlNet
# 0. 环境



# 1. model

# 2.data

# 3. train
- ControlNet/ldm/models/diffusion/ddpm.py

- base
    - DDPM 46行
        - base class
        - 大部分关于DM的sample 公式计算
    - LatentDiffusion(DDPM) 523行
        - 多一个autoencoder 
    - DiffusionWrapper 1320行
        - 特征融合的方式