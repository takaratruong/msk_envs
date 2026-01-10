# MSK Envs - RL Environments for Musculoskeletal Simulations
RL environments built on top of the [MSK Warp](https://github.com/willwng/msk_warp) simulator that run entirely on the GPU.

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
    
To start a training run, use the corresponding config (found in [hyperparams](msk_envs/train/hyperparams.py)):
```bash
python -m msk_envs.train.train [sprint|vertical|walk] --exp_prefix my_training_run
```

Or to launch several runs on a SLURM cluster:
```
python slurm/deploy.py --input_yaml slurm/cfg/baselines.yaml --mode gen_run
```

### Motion imitation environment
To track a motion file (in this example, the starting-phase of a sprint):
```bash
python -m msk_envs.train.train imitate --exp_prefix my_imitation --motion_name "../motions/study2_p02_s_01_lowIK.mot"
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
