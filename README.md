
# Setup

```
conda create -n msk python=3.11
conda activate msk
pip install torch torchvision
pip install -r requirements.txt
cd ../msk_warp/  # git clone git@github.com:willwng/msk_warp.git
pip install -e .
```


# Training

## Sprint env

```
python -m msk_envs.train.fasttd3.train
```

Or with SLURM:
```
python slurm/deploy.py --input_yaml slurm/cfg/baselines.yaml --mode gen_run
```

# Visualization

### Dashboard
During training, every `eval_freq` steps, the training script will save a 
a checkpoint and a trajectory json. This trajectory can be visualized with 
the web-viewer.
```bash
cd dashboard
python3 dashboard.py
```
These trajectories can also be loaded into Blender. Open Blender and in the 
scripting tab open `scripts/madrona-env/blender.py`, edit the trajectory path,
and run the script (it may take a minute or so). It should set up all the 
animation frames.