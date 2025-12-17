
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

To start a training run,
```bash
python -m msk_envs.train.fasttd3.train --env-variant [SPRINT|VERTICAL|WALK] --exp_prefix my_training
```

Or with SLURM:
```
python slurm/deploy.py --input_yaml slurm/cfg/baselines.yaml --mode gen_run
```

# Motion imitation

## Parse motion to be imitated
```bash
python -m msk_envs.utils.parse_mot --motion msk_envs/motions/reference_stride.mot
```

## Train
```bash
python -m msk_envs.train.fasttd3.train --env-variant IMITATE --exp_prefix my_imitation --motion_name reference_stride
```

# Visualization

### Dashboard
During training, every `eval_freq` steps, the training script will save a 
checkpoint and a trajectory JSON. This trajectory can be visualized with 
the web-viewer.
```bash
cd dashboard
python3 dashboard.py
```