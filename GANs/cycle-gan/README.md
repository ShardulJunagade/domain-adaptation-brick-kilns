# CycleGAN

A clean, simple and readable implementation of CycleGAN in PyTorch.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)


### Training
Horse and Zebras Dataset: [Link](https://www.kaggle.com/datasets/suyashdamle/cyclegan)

Create a folder named "dataset" with "train" and "test" subfolders, each containing "horses" and "zebras" subfolders. The structure:

```bash
root
├── config.py
├── dataset.py
├── discriminator.py
├── generator.py
├── train.py
├── utils.py
└── dataset
    ├── train
    │   ├── horses
    │   └── zebras
    └── test
        ├── horses
        └── zebras

```


Edit the `config.py` file to match the setup you want to use. Then run `train.py`.


### Download pretrained weights
Pretrained weights download.

Extract the zip file and put the pth.tar files in the directory with all the python files. Make sure you put LOAD_MODEL=True in the `config.py` file.
