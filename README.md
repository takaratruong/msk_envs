
# Install

```
conda create -n msk python=3.11
conda activate msk
pip install torch torchvision
pip install -r requirements.txt
cd ../msk_warp/  # git clone git@github.com:willwng/msk_warp.git
pip install -e .
```


# Train

## Sprint env

```
python -m msk_envs.train.fasttd3.train
```

Or with SLURM:
```
python slurm/deploy.py --input_yaml slurm/cfg/baselines.yaml --mode gen_run
```