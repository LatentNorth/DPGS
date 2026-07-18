[README.md](https://github.com/user-attachments/files/30152061/README.md)
# DPGS

This repository contains the evaluation code for DPGS under balanced and Dirichlet-imbalanced transductive few-shot learning settings. The current configuration uses pre-extracted WRN-28-10 (S2M2) features.

## Download Features

The pre-extracted WRN-28-10 (S2M2) features can be downloaded from [Google Drive](https://drive.google.com/file/d/1RO1X989fK97ChSfPdhjDFccyu8FWUpZy/view?usp=sharing). These features follow the package used by [PUTM](https://github.com/RashLog/PUTM) and are based on the [S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot) backbone.

After downloading, extract the `features` folder and place it directly in the repository root:

```text
DPGS/
└── features/
    └── wrn_s2m2/
        ├── cifar/
        │   └── novel.plk
        ├── cub/
        │   └── novel.plk
        ├── mini/
        │   └── novel.plk
        └── tiered/
            └── novel.plk
```

Do not create an extra nested directory such as `DPGS/features/features/`.

## Code Structure

```text
DPGS/
├── cache/                              # Created locally; stores random task states
├── config/
│   ├── base_config.yaml
│   ├── balanced/
│   │   └── methods_config/
│   │       └── dpgs.yaml
│   └── dirichlet/
│       └── methods_config/
│           └── dpgs.yaml
├── features/
│   └── wrn_s2m2/
│       ├── cifar/
│       ├── cub/
│       ├── mini/
│       └── tiered/
├── methods/
│   └── dpgs.py
├── eval.py
├── FSLTask_im.py
├── utils.py
└── README.md
```

### About the `cache` Directory

Git tracks files rather than empty directories, so an empty `cache/` folder will not appear after uploading or cloning the repository. Create it manually from the repository root before the first evaluation:

```bash
mkdir cache
```

The directory must be writable. DPGS uses it to save and reload random task states for reproducible evaluation.

## Requirements

The code requires Python 3 and the following main packages:

- PyTorch
- NumPy
- SciPy
- scikit-learn
- PyYAML
- tqdm

## Evaluation

Run all commands from the repository root. First, edit `config/base_config.yaml` to select the dataset, task count, shots, and evaluation distribution.

### Dirichlet-Imbalanced Evaluation

Set the following value in `config/base_config.yaml`:

```yaml
balanced: 'dirichlet'
```

Then run:

```bash
python eval.py --base_config config/base_config.yaml --method_config config/dirichlet/methods_config/dpgs.yaml
```

### Balanced Evaluation

Set the following value in `config/base_config.yaml`:

```yaml
balanced: 'balanced'
```

Then run:

```bash
python eval.py --base_config config/base_config.yaml --method_config config/balanced/methods_config/dpgs.yaml
```

The distribution value in `base_config.yaml` must match the directory used by `--method_config`.

## Results

Evaluation progress and mean accuracy are printed in the terminal. A summary containing accuracy, F1 score, and AUC is appended to `results.txt` in the repository root.
