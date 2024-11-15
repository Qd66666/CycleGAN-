# Cross-Domain Visual Loop Closure Detection Based on Image Translation Model


## Setup

### Prerequisites
- Pytorch
- Linux 
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

You can see the domain-invariant images at .../result/images/{dataset_name}/CycleGAN+/{Epoch_num}, after the test steps.

Here is an example of the domain-invariant images:

![Image text](https://github.com/Qd66666/Improve-CycleGAN/blob/8103d546363b4181f3750e7d7f36952438990263/img_240.png)

## Similarity calculation

### Installation

Download and move dependencies as described in the must_downloads.txt file.

### Calculation

Divide the images into reference folder and query folder. Then modify the query_dir and ref_dir parameters in the Alexnet.py/AmosNet.py file and run the file. The generated similarity matrix will be saved in the file '/{result_martix_model_name.txt_dir}'.
