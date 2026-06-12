# MSK Envs - RL Environments for Musculoskeletal Simulations
RL environments built on top of the [Bolt](https://github.com/willwng/bolt) simulator that run entirely on the GPU.

## Preqrequisites/Setup
Installation requires installation of [Bolt](https://github.com/willwng/bolt).
```
cd "path to bolt"  # git clone git@github.com:willwng/bolt.git
pip install -e .
```
And rest of packages:
```
cd "path to msk_envs"
pip install -r requirements.txt
```

## Training
We provide example training code based on [FastTD3](https://github.com/younggyoseo/FastTD3), including hyperparameters for the following example environments.

### Sprinting (maximum forward velocity) environment
    
To start a training run, use the corresponding config (found in [hyperparams](msk_envs/train/hyperparams.py)):
```bash
python -m msk_envs.train.train [sprint|vertical|walk] --exp_prefix my_training_run --algo td3 env_config:sprinter 
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
