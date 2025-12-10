import argparse
import yaml
import os
from copy import deepcopy


def underscore2camelCase(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


def split_exp(exp_name, exp_d):
    had_split = False
    ret_dict = {}
    for k, v in exp_d.items():
        if type(v) == list:
            had_split = True
            break
    if had_split:
        for item in v:
            _added_dict = deepcopy(exp_d)
            _added_dict[k] = item
            if k[-1] != 'model_path':
                item_str = item
            else:
                path, file = os.path.split(item)
                model_no_suffix = os.path.splitext(file)[0].replace('model', '')
                exp = os.path.split(path)[1]
                item_str = exp + '_' + model_no_suffix
            ret_dict.update({f'{exp_name}_{underscore2camelCase(k[-1])}_{item_str}': _added_dict})

    return had_split, ret_dict


def split_experiments(exp_dicts):
    # keys: exp names
    # values: flatten yaml dicts
    # in-place
    had_split = True
    while had_split:
        had_split = False  # init flag
        for exp_name, exp_d in exp_dicts.items():
            had_split, splited_exp = split_exp(exp_name, exp_d)
            if had_split: break
        if had_split:
            del exp_dicts[exp_name]
            exp_dicts.update(splited_exp)


def flatten(dictionary, cur_key=()):
    output = dict()
    for k, v in dictionary.items():
        if isinstance(v, dict):
            output.update(flatten(v, cur_key=cur_key + (k,)))
        else:
            output[cur_key + (k,)] = v
    return output


def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        d = resultDict
        for part in key[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[key[-1]] = value
    return resultDict


def build_experiments(yaml_dict, exp_name):
    yaml_flat = flatten(yaml_dict)
    exps = {underscore2camelCase(exp_name): yaml_flat}
    split_experiments(exps)
    return {k.replace(' ', '_'): unflatten(v) for k, v in exps.items()}

def yaml2cmd(exp_name, exp_dict):
    cmd = exp_dict['base_cmd']
    if '--exp-prefix' in exp_dict['base_cmd']:
        cmd += f" {exp_name}"  # name is automatically generated
    if 'eval.' in exp_dict['base_cmd']:
        cmd += f' --eval_name {exp_name}'  # name is automatically generated
    for arg, val in exp_dict['args'].items():
        cmd += f' --{arg} {val}'
        if arg == 'eval_mode' and val == 'inversion' and 'guidance_param' not in exp_dict['args'].keys():
            cmd += f' --guidance_param 1.0'
    if 'store_trues' in exp_dict.keys():
        for arg, val in exp_dict['store_trues'].items():
            if val:
                cmd += f' --{arg}'

    return cmd

def build_cmds(yaml_dict, exp_name):
    yamls = build_experiments(yaml_dict, exp_name)
    cmds = [yaml2cmd(name, params) for name, params in yamls.items()]
    return cmds, yamls

def generate_experiments(opt):
    assert opt.input_yaml.endswith('.yaml')
    exp_name = os.path.basename(opt.input_yaml).replace('.yaml', '')
    exp_dir = os.path.join(opt.out_dir, exp_name)
    assert not os.path.exists(exp_dir), f'[{exp_dir}] already exists.'

    # generate yaml per experiment
    with open(opt.input_yaml, 'r') as fr:
        top_yaml = yaml.safe_load(fr)
    # exps = build_experiments(top_yaml, exp_name)
    cmd_lines, exp_dict = build_cmds(top_yaml, exp_name)

    # save yaml and bash files
    os.makedirs(exp_dir)
    with open(opt.sbatch_template, 'r') as fr:
        sh_template = fr.read()
    for _name, _cmd, e_dict in zip(exp_dict.keys(), cmd_lines, exp_dict.values()):
        # yaml_path = os.path.join(exp_dir, f'{e_name}.yaml')
        # sh_path = yaml_path.replace('.yaml', '.sh')
        slurm_path = os.path.join(exp_dir, f'{_name}.slurm')
        sh_content = sh_template.replace('PATH', slurm_path.replace('.slurm', '')) + '\n' + _cmd

        with open(slurm_path, 'w') as fw:
            fw.writelines(sh_content)

    print(f'SLURM scripts were generated in [{exp_dir}]')
    return


def run_experiments(opt):
    assert opt.input_yaml.endswith('.yaml')
    exp_name = os.path.basename(opt.input_yaml).replace('.yaml', '')
    exp_dir = os.path.join(opt.out_dir, exp_name)
    assert os.path.exists(exp_dir)

    slurm_list = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('.slurm')]
    for sh_file in slurm_list:
        task = f'sbatch {sh_file}'
        os.system(task)
        print(f'Sent task [{task}]')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_yaml', type=str, required=True,
                        help='Input yaml file, parameters to sweep will be defined with tuples')
    parser.add_argument('--out_dir', type=str, default='slurm_exp', help='bash run files will be written to here.')
    parser.add_argument('--sbatch_template', type=str, default='./slurm/template.slurm',
                        help='bash run files will be written to here.')
    parser.add_argument('--mode', type=str, default='gen_only', choices=['gen_only', 'run_only', 'gen_run'],
                        help='Specify either gen/run or both')
    opt = parser.parse_args()
    if 'gen' in opt.mode:
        generate_experiments(opt)
    if 'run' in opt.mode:
        run_experiments(opt)