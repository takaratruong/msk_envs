# MSK Envs - RL Environments for Musculoskeletal Simulations
MSK Envs are environments designed for training RL policies. Environments run entirely on a GPU and interoperate with the simulator built on top of the hardware-accelerated musculoskeletal simulator [MSK Warp](https://github.com/willwng/msk_warp). 

## Preqrequisites/Setup
Installation requires installation of [MSK Warp](https://github.com/willwng/msk_warp), of which the only dependency is [warp-lang](https://github.com/NVIDIA/warp). Use of the environments requires PyTorch. 
```
conda create -n msk python=3.11
conda activate msk
pip install warp-lang
pip install torch torchvision
pip install -r requirements.txt
cd ../msk_warp/  # git clone git@github.com:willwng/msk_warp.git
pip install -e .
```


## Training
We provide example training code based on [FastTD3](https://github.com/younggyoseo/FastTD3), including hyperparameters for the following example environments.

### Sprinting (maximum forward velocity) environment
    
To start a training run,
```bash
python -m msk_envs.train.fasttd3.train --env-variant [SPRINT|VERTICAL|WALK] --exp_prefix my_training
```

Or with SLURM:
```
python slurm/deploy.py --input_yaml slurm/cfg/baselines.yaml --mode gen_run
```

### Motion imitation environment
    
#### Parse motion to be imitated
```bash
python -m msk_envs.utils.parse_mot --motion msk_envs/motions/reference_stride.mot
```

#### Train
```bash
python -m msk_envs.train.fasttd3.train --env-variant IMITATE --exp_prefix my_imitation --motion_name reference_stride
```


## Visualization
We provide visualization tools (as a dashboard/web-viewer) as well as scripts to generate analytics of trajectories (as PDFs).
### Dashboard
During training, every `eval_freq` steps, the training script will save a 
checkpoint and a trajectory JSON. This trajectory can be visualized with 
the web-viewer.
```bash
cd dashboard
python3 dashboard.py
```

#### Visualize from remote

First, mount the remote traj dir to your local machine

```
sshfs sc.stanford.edu:/move/u/guytevet/msk_envs/dashboard/trajectories ./dashboard/remote_trajectories
```

Then run the dashboard pointing to it
```
cd dashboard
python dashboard.py --traj-dir remote_trajectories
```
