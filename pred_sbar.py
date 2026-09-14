"""Replay a saved S_bar best model on its complete original MAT dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from constitutive_pinn.config import ModelArchitecture as Mode
from constitutive_pinn.model import ModularConstitutivePINN
from constitutive_pinn.sbar_data import load_sbar_data
from constitutive_pinn.training import select_device
from train_sbar import export


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args(arguments)
    if args.batch_size <= 0:
        raise ValueError('Batch size must be positive')
    torch.set_num_threads(4)
    device = select_device(args.device)
    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'sbar_v1' or checkpoint.get('stress_measure') != 'S_bar':
        raise ValueError('Expected an S_bar checkpoint from train_sbar.py')
    stage = Mode.parse(checkpoint['architecture'])
    if stage.use_volumetric:
        raise ValueError('S_bar does not support the volumetric branch')
    data = load_sbar_data(checkpoint['data_path'])
    for field in ('directions', 'weights'):
        if not np.array_equal(checkpoint[field], getattr(data, field)):
            raise ValueError('Checkpoint and data quadrature differ')
    train_idx, val_idx = checkpoint['train_indices'], checkpoint['validation_indices']
    if not np.array_equal(np.sort(np.r_[train_idx, val_idx]), np.arange(len(data.current))):
        raise ValueError('Saved split differs from dataset length')
    state = checkpoint['state_dict']
    model = ModularConstitutivePINN(data.directions, data.weights,
            float(state['x_mean']), float(state['x_std']), architecture=stage).to(device)
    model.load_state_dict(state, strict=True)
    current = torch.as_tensor(data.current, device=device)
    maximum = torch.as_tensor(data.maximum, device=device)
    output = Path(args.output_dir) if args.output_dir else path.parent/'prediction'
    output.mkdir(parents=True, exist_ok=True)
    history_path = path.parent/'history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    print(f'Best epoch {checkpoint["best_epoch"]}; predicting all {len(current)} original samples', flush=True)
    export(model, data, current, maximum, train_idx, val_idx, output, history, args.batch_size)


if __name__ == '__main__':
    main()
