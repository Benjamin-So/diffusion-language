# Diffusion Reward on Language Environment

This repository implements the Diffusion Reward pipeline on the Google Language Table environment

Set the Python Path in root directory
```
cd diffusion_language
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
## Set up Diffusion_Reward environment:

```bash
cd diffusion_reward
export PYTHONPATH=${PWD}:$PYTHONPATH
```
Create a virtual environment.
```bash
conda env create --name diffusion_language -f conda_env.yml 
conda activate diffusion_reward
pip install -e .
```
Install extra dependencies.
- Install PyTorch.
```bash
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
Hack to fix failed to initialize OpenGL error
```
unset LD_PRELOAD
```

## Set up Language_Table environment:
```
cd language_table
pip install -r ./requirements_static.txt
export PYTHONPATH=${PWD}:$PYTHONPATH
```
## Training
Run the RL training loop for the blocktoblock64 task, which used VQ-VAE to train on 64x64 frames of 'move the green star to the blue cube'
```
bash scripts/run/rl/drqv2_langauge_diffusion_reward.sh blocktoblock64
```
