# GMFnet
This repository contains the implementation of [ConvGRU-based Multi-scale Frequency Fusion Network for PAN-MS Joint Classification](https://ieeexplore.ieee.org/abstract/document/10570233) (IEEE Transactions on Geoscience and Remote Sensing). 

## Overall Architecture
<img src='https://github.com/Xidian-AIGroup190726/GMFnet/blob/main/src/Overall%20structure.png' style="zoom:50%;"/>

## Visualization results of relevant features for GGSFM
<img src='https://github.com/Xidian-AIGroup190726/GMFnet/blob/main/src/decompose.png'>

## Run test.py to use GMFnet
```
import torch
from model.gmfnet import Net 

if __name__ == '__main__':
    ms = torch.randn([1, 4, 16, 16).to(device)
    pan = torch.randn([1, 1, 64, 64]).to(device)
    args = {
        'num_channels': 4,
        'patch_size': 32,
        'device': device
    }
    module = Net(args).to(device)
    result = module(ms, pan)
    print(result.shape)   # torch.size([B, C, H, W])
```

## Project code
The complete project code is available at [https://github.com/salalalala23/Dual-modal-fusion](https://github.com/salalalala23/Dual-modal-fusion).

## Citation
Please cite GMFnet in your publications if it helps your research:
```
@ARTICLE{10570233, 
  author={Zhu, Hao and Yi, Xiaoyu and Li, Xiaotong and Hou, Biao and Jiao, Changzhe and Ma, Wenping and Jiao, Licheng}, 
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ConvGRU-Based Multiscale Frequency Fusion Network for PAN-MS Joint Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Feature extraction;Transforms;Remote sensing;Logic gates;Long short term memory;Data mining;Satellites;Contourlet transform (CT);ConvGRU;deep learning;remote sensing},
  doi={10.1109/TGRS.2024.3415371}}
```
