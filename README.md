# Diffusion Reward on Language Environment\
Set the Python Path in root directory
```
cd diffusion_language \
export PYTHONPATH=$PYTHONPATH:$(pwd)\
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
- Install mujoco210 and mujoco-py following instructions [here](https://github.com/openai/mujoco-py#install-mujoco).

- Install Adroit dependencies.
```bash
cd env_dependencies
pip install -e mj_envs/.
pip install -e mjrl/.
cd ../..
```
- Install MetaWorld following instructions [here](https://github.com/Farama-Foundation/Metaworld?tab=readme-ov-file#installation).

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
