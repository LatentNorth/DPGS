import random
import argparse
import datetime
import numpy as np
import torch
import FSLTask_im

from methods.dpgs import DPGS

from utils import load_cfg_from_cfg_file, merge_cfg_from_list


def parse_args() -> argparse.Namespace:
    """Load and merge base, method, and command-line configuration."""
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config',  type=str, required=True, help='Method config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    method_config = args.method_config
    assert args.base_config is not None
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    for arg in args.__dict__.keys():
        if arg not in ['base_config', 'method_config', 'opts']:
            cfg.update({arg: args.__dict__[arg]})
    assert method_config.split("/")[1] == cfg.balanced
    return cfg


def centerDatas(datas):
    """Center every task around its mean feature vector."""
    datas[:] -= datas.mean(1, keepdim=True)
    return datas


def scaleEachUnitaryDatas(datas):
    """L2-normalize every feature vector."""
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    """Apply task-wise QR reduction."""
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1), 'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


def fix_seed(seed):
    """Seed Python, NumPy, and PyTorch for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_method_builder(model, device, log_file, args):
    """Build the DPGS evaluator."""
    if args.method != 'DPGS':
        raise ValueError("This evaluator only supports method='DPGS'")

    method_info = {'model': model, 'device': device, 'log_file': log_file, 'args': args}
    print("DPGS method selected!")
    return DPGS(**method_info)


def compute_confidence_interval(data, axis=0):
    """Return the mean and its 95% confidence interval."""
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm


def get_tasks(n_ways, n_shot, n_queries, n_runs, backbone, dataset, distribution):
    """Generate a batch of balanced or Dirichlet few-shot tasks."""
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries, 'tasks': n_runs, 'sample': distribution}

    FSLTask_im.loadDataSet(backbone, dataset)
    FSLTask_im.setRandomStates(cfg)

    ndatas, labels, query_samples = FSLTask_im.GenerateRunSet(cfg=cfg)

    if cfg['sample'] == 'uniform':
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
        labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,                                                                                                    n_samples)
    elif cfg['sample'] == 'dirichlet':
        pass
    print("size of the datas...", ndatas.size())
    return ndatas, labels


if __name__ == '__main__':
    fix_seed(2)

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_ways = args.n_ways
    n_queries = args.n_query
    n_runs = args.number_tasks

    # Configure reproducible task sampling.
    FSLTask_im._maxRuns = n_runs
    FSLTask_im._alpha = args.alpha_dirichlet

    backbone = args.arch
    dataset = args.dataset

    distribution = 'dirichlet' if args.balanced == 'dirichlet' else 'uniform'

    print("backbone:{}, dataset:{}, distribution:{}, dirichlet:{}".format(backbone, dataset, distribution, args.alpha_dirichlet))

    for n_shot in args.shots:
        args.__dict__.update({'shot': n_shot})
        ndatas, labels = get_tasks(n_ways, n_shot, n_queries, n_runs, backbone, dataset, distribution)
        ndatas = ndatas.to(device)
        labels = labels.to(device)

        model = None

        method = get_method_builder(model, device, None, args)

        logs = method.run_task(ndatas, labels, args)

        # Summarize per-task metrics.
        acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
        f1_mean, f1_conf = compute_confidence_interval(logs['f1'][:, -1])
        auc_values = logs['auc'][:, -1]
        auc_values = auc_values[~np.isnan(auc_values)]

        if len(auc_values) > 0:
            auc_mean, auc_conf = compute_confidence_interval(auc_values)
        else:
            auc_mean, auc_conf = np.nan, np.nan


        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append one row for the current evaluation setting.
        with open("results.txt", "a+") as f:
            f.write(f'{current_time},{dataset}, {args.method}, {n_shot}, {distribution}, {args.alpha_dirichlet}, {acc_mean:.4f}±{acc_conf:.4f},{f1_mean:.4f},{auc_mean:.4f}\n')
