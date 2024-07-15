### Diffusion Reward Simple ReadMe\
Set the Python Path
```
cd diffusion_language \
export PYTHONPATH=$PYTHONPATH:$(pwd)\
cd diffusion_reward\
export PYTHONPATH=$PYTHONPATH:$(pwd)\
cd ../language_table\
export PYTHONPATH=$PYTHONPATH:$(pwd) \
```
Run the RL training loop for the blocktoblock64 task, which used VQ-VAE to train on 64x64 frames of 'move the green star to the blue cube'
```
bash scripts/run/rl/drqv2_langauge_diffusion_reward.sh blocktoblock64
```
