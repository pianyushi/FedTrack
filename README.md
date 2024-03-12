# FedTrack

## Environment

- pyqmdnet.yaml

- pysot-tookit/requirements.txt

## Introduction

PyTorch implementation of FedTrack, which runs with a single CPU core and a single GPU (GTX 3080 Ti).

## Datasets

- training datasets: **Imagenet-VID and vot2018\19\21**

- testing datasets: **OTB100, LaSOT, GOT-10k**

## Results on OTB

| ![](README_md_files/3a680620-e052-11ee-9f02-d176b0571c05.jpeg?v=1&type=image) | ![](README_md_files/3d823e70-e052-11ee-9f02-d176b0571c05.jpeg?v=1&type=image) |
| :---------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |

## Prerequisites

- python 3.9+

- opencv 4.6+

- [PyTorch 2.0+](http://pytorch.org/) and its dependencies

- for GPU support: a GPU with \~4G memory

## Usage

### Co-Tracking Training

```bash
 python pretrain/fedtrain.py
```

### Evaluation

- Visualization of the trace

       python tracking/run_tracker.py -s Basketball [-d (display fig)] [-f (save fig)]

  - You can provide a sequence configuration in two ways (see tracking/gen_config.py):

    - `python tracking/run_tracker.py -s [seq name]`

    - `python tracking/run_tracker.py -j [json path]`&#x20;

- Draw success and precision

  ```bash
   python pysot-tookit/pysot/visualization/draw_success_precision.py
  ```

- Pretraining on ImageNet-VID

  - Download [ImageNet-VID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid) dataset into "datasets/ILSVRC"

  ```bash
   python pretrain/prepro_imagenet.py
   python pretrain/train_mdnet.py -d imagenet
  ```
