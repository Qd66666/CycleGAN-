## Cross-Domain Visual Loop Closure Detection Based on Image Translation Model


## Setup

### Getting Started
- Pytorch
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Training


usage:
```shell
python train.py --dataroot ./{dataroot_dir} --name CycleGAN+ --gpu_ids {GPU_NUM} --no_dropout --no_AA_BB
```
## Testing

usage:
```shell
python test.py --epoch {Epoch_num} --gpu_ids {GPU_NUM} --name CycleGAN+
  --dataroot ./{dataroot_dir} --no_dropout --phase train --eval
```

## Visualization

You can see the domain-invariant images at .../result/images/{dataset_name}/{model_name}/{Epoch_num}, after the test steps.
Here is a example of the domain-invariant images

![Image text](https://github.com/Qd66666/Improve-CycleGAN/blob/8103d546363b4181f3750e7d7f36952438990263/img_240.png)






# Multispectral Domain Invariant Image for Retrieval-based Place Recognition
- [ICRA2020 Paper](./MDII_paper.pdf)
- [ICRA2020 Presentation](https://www.slideshare.net/SejongRCV/multispectral-domain-invariant-image-for-retrievalbased-place-recognition-234803884)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started


### Installation

- Clone Repo

```sh
git clone https://github.com/sejong-rcv/MDII
cd MDII
```
### Docker 

- Prerequisite 
  - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 
- Option
  - visdom port number
   
```sh
nvidia-docker run -it -v $PWD:/workspace -p {port}:8888 -e NVIDIA_VISIBLE_DEVICES=all handchan/mdii /bin/bash
```
> if you have any problem with downloading the docker image, check this repository : https://hub.docker.com/r/handchan/mdii/tags

### Dataset

- Download Dataset

```sh
cd MDII
curl http://multispectral.sejong.ac.kr/ICRA2020_MDII/ICRA_MDII.tar.gz -o ICRA_MDII.tar.gz
tar -xzvf ICRA_MDII.tar.gz
```

- we support the pre-processed dataset.If you want to check the original dataset, refer to the following papers.
  - [All-day Vision Dataset (CVPRW 2015)](https://sites.google.com/site/ykchoicv/multispectral_vprice)
  - [All-day Vision Dataset (TITS 2018)](https://ieeexplore.ieee.org/document/8293689)

### Train

- Running train.py 

```sh
python train.py --name MDII --model MDII_gan --dataroot ./ICRA_MDII --gpu_ids GPU_NUM  --no_dropout --no_AA_BB
```

### Convert
- Running feat_c.py , make .npz file

```sh
### Convert train img to MDII
python feat_c.py --epoch {Epoch} --gpu_ids {GPU_NUM} --name MDII \ 
  --dataroot ./ICRA_MDII --no_dropout --model MDII_gan --phase train --eval
### Convert test img to MDII
python feat_c.py --epoch {Epoch} --gpu_ids {GPU_NUM} --name MDII \
  --dataroot ./ICRA_MDII --no_dropout --model MDII_gan --phase test --eval
```

### Evaluation
- Using Matlab vlfeat code. run rank.py
  - Download [VLFeat](https://www.vlfeat.org/) (our version is vlfeat-0.9.21)
  - Replace {vlfeat dir}/apps/recognition/ to [recognition_MDII](./recognition/)
  - Place your convert result name as {vlfeat dir}/MDII
    ```sh
    cd {vlfeat dir}
    ln -s {result dir} MDII (ex. ../../result/images/ICRA_MDII/{checkpoint name}/{epoch}/)
    # {vlfeat dir}
    # ├── apps
    # │   └── recognition
    # ├── data
    # │   ├── MDII -> ../../result/images/ICRA_MDII/{checkpoint name}/{epoch}/
    # │   │   ├── test
    # │   │   │   ├── rgb
    # │   │   │   ├── thr
    # │   │   ├── train
    # │   │   │   ├── rgb
    # │   │   │   ├── thr
    # ├──
    # ...
    ```
   - Run the Matlab code {vlfeat dir}/apps/recognition/experiments.m
   - Run the python code rank.py {workspace/rank.py}
   ```sh
   python rank.py --cache_path ./{vlfeat dir}/data_MDII_0604_200epoch/ex-MDII-vlad-aug
   # You can see the detail in python rank.py --help
   ```
### Citation

```
@INPROCEEDINGS{ICRA2020,
  author = {Daechan Han*, YuJin Hwang*, Namil Kim, Yukyung Choi},
  title = {Multispectral Domain Invariant Image for Retrieval-based Place Recognition},
  booktitle = {International Conference on Robotics and Automation(ICRA)},
  year = {2020}
}
```
