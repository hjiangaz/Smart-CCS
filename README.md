<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Media\logos\smart-ccs_dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="Media\logos\smart-ccs_light.jpg">
  <img alt="Smart-CCS Logo" src="Media\logos\smart-ccs_light.jpg">
</picture>

[![arXiv](https://img.shields.io/badge/arXiv-2502.09662-%23B31B1B.svg)](https://www.arxiv.org/abs/2502.09662) [![Python Version](https://img.shields.io/badge/Python-3.9.0-green.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Demo Video](https://img.shields.io/badge/Demo-Video-%23FF0000.svg)](https://www.youtube.com)

This is the official repository of  
**Generalizable Cervical Cancer Screening via Large-scale Pretraining and Test-Time Adaptation**  
by _Hao Jiang, Cheng Jin, Huangjing Lin, Yanning Zhou, Xi Wang, Jiabo Ma, Li Ding, Jun Hou, Runsheng Liu, Zhizhong Chai, Luyang Luo, Huijuan Shi, Yinling Qian, Qiong Wang, Changzhong Li, Anjia Han, Ronald Chan, Hao Chen_

---

## Installation

### Requirements

- **Python 3.9.0**
- CUDA 12.2
- PyTorch 2.0.0

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/hjiangaz/Smart-CCS
```

2. **Install the liabarary**

```bash
cd subfolders
pip install -r requirements.txt
```

3. **Download pretrained weights**

Download [pretrained models](https://drive.google.com/drive/folders/1KbYIU5AjbTG8kIG-CjLM5aftvQtkgib3?usp=sharing), including feature extractor, detector, and classifier

### QuickStart

1. **Cell-level Screening**

```bash
cd Det-cell
sh ./configs/extract.sh
```

2. **WSI-level Screening**

```bash
cd Cls-WSI
sh ./scripts/test.sh
```
### License

This project is covered under the Apache 2.0 License.

### Citation

If you find this work useful, please cite:

```bash
@article{jiang2025generalizable,
  title={Generalizable Cervical Cancer Screening via Large-scale Pretraining and Test-Time Adaptation},
  author={Jiang, Hao and Jin, Cheng and Lin, Huangjing and Zhou, Yanning and Wang, Xi and Ma, Jiabo and Ding, Li and Hou, Jun and Liu, Runsheng and Chai, Zhizhong and others},
  journal={arXiv preprint arXiv:2502.09662},
  year={2025}
}
```
