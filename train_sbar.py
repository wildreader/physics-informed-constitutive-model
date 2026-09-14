"""Train S_bar using the supplied Xtrain/Ytrain, then predict every original row.

Examples:
  python train_sbar.py --mode no_damage
  python train_sbar.py --mode damage --elastic-checkpoint outputs/sbar/no_damage/model.pth
  python train_sbar.py --mode both
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import torch
from sklearn.model_selection import train_test_split
from constitutive_pinn.config import ModelArchitecture as Mode
from constitutive_pinn.model import ModularConstitutivePINN
from constitutive_pinn.data import component_rms_scale
from constitutive_pinn.sbar_data import load_sbar_data, assert_sbar_pair
from constitutive_pinn.training import set_reproducible_seed, select_device


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['no_damage', 'damage', 'both'], default='both')
    p.add_argument('--no-damage-data', default='proportional_biaxial_Sbar_NoDamage.mat')
    p.add_argument('--damage-data', default='proportional_biaxial_Sbar.mat')
    p.add_argument('--elastic-checkpoint', default=None)
    p.add_argument('--output-dir', default='outputs/sbar')
    p.add_argument('--free-energy-epochs', type=int, default=500)
    p.add_argument('--damage-epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=0.2)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--progress-interval', type=int, default=10)
    return p


def predict(model, current, maximum, batch_size):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(current), batch_size):
            parts.append(model.forward_sbar(current[start:start+batch_size],
                                           maximum[start:start+batch_size]).detach().cpu().numpy())
    return np.concatenate(parts)


def metrics(predicted, target):
    error = predicted.astype(np.float64) - target
    rms = np.sqrt(np.mean(np.square(target.astype(np.float64)), axis=0))
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return {'rmse_components': rmse.tolist(),
            'normal_relative_l2': float(np.linalg.norm(error[:, :3]) / max(np.linalg.norm(target[:, :3]), 1e-12)),
            'normal_component_relative_l2': (rmse[:3] / np.maximum(rms[:3], 1e-12)).tolist(),
            'max_absolute_error': np.max(np.abs(error), axis=0).tolist()}


def export(model, data, current, maximum, train_idx, val_idx, directory, history, batch_size):
    pred = predict(model, current, maximum, batch_size)
    reports = {name: metrics(pred[ix], data.target[ix]) for name, ix in
               [('all', np.arange(len(pred))), ('train', train_idx), ('validation', val_idx)]}
    reports['cases'] = {str(int(c)): metrics(pred[data.case_index == c], data.target[data.case_index == c])
                        for c in np.unique(data.case_index)}
    (directory / 'metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    savemat(directory / 'full_prediction.mat', {
        'Sbar_pred': pred.T, 'Sbar_true': data.target.T,
        'sample_indices_matlab': np.arange(1, len(pred)+1),
        'train_indices_matlab': train_idx+1, 'validation_indices_matlab': val_idx+1,
        'lambdaX': data.lambda_x, 'caseIndex': data.case_index,
        'biaxialRatio': data.ratios, 'stress_measure': 'S_bar',
        'voigt_order': '11,22,33,12,13,23'})
    for c in np.unique(data.case_index):
        ix = np.flatnonzero(data.case_index == c)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for k, ax in enumerate(axes.flat):
            ax.plot(data.lambda_x[ix], data.target[ix, k], color='black', lw=1, label='MAT target')
            ax.plot(data.lambda_x[ix], pred[ix, k], color='tab:red', lw=1, linestyle='--', label='Best model')
            ax.set(xlabel='lambda_x', ylabel='S_bar '+['11','22','33','12','13','23'][k])
            ax.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(f'Case {int(c)} / biaxial ratio {data.ratios[ix[0]]:g} / original path order')
        fig.savefig(directory / f'case_{int(c)}.png', dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.semilogy([h['train_loss'] for h in history], label='Training batches')
    ax.semilogy([h['validation_loss'] for h in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='RMS-scaled S_bar MSE')
    ax.legend()
    fig.savefig(directory / 'loss.png', dpi=140)
    plt.close(fig)
    print('Full/validation normal relative L2:', reports['all']['normal_relative_l2'],
          reports['validation']['normal_relative_l2'], flush=True)


def fit_stage(model, data, stage, args, train_idx, val_idx, directory, source_path):
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    current, maximum, target = [torch.as_tensor(a, device=device) for a in
                                (data.current, data.maximum, data.target)]
    scale = torch.as_tensor(component_rms_scale(data.target[train_idx]), device=device)
    train_ids = torch.as_tensor(train_idx, device=device)
    val_ids = torch.as_tensor(val_idx, device=device)
    model.set_training_stage(stage)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()
              if k.startswith('free_energy_module.') and stage is Mode.FREE_ENERGY_DAMAGE}
    optimizer = torch.optim.Adam(model.stage_parameters(stage), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    epochs = args.free_energy_epochs if stage is Mode.FREE_ENERGY else args.damage_epochs
    best, best_state, history = float('inf'), None, []
    start = time.monotonic()
    for epoch in range(1, epochs+1):
        model.train()
        order = train_ids[torch.randperm(len(train_ids), device=device)]
        total = 0.0
        for ix in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_sbar(current[ix], maximum[ix])
            loss = (((prediction-target[ix])/scale)**2).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach().item()*len(ix)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for ix in val_ids.split(args.batch_size):
                prediction = model.forward_sbar(current[ix], maximum[ix])
                loss = (((prediction-target[ix])/scale)**2).mean()
                val_total += loss.detach().item()*len(ix)
        train_loss, val_loss = total/len(train_ids), val_total/len(val_ids)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError('Nonfinite training loss')
        history.append({'epoch':epoch, 'train_loss':train_loss, 'validation_loss':val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'format':'sbar_v1', 'stress_measure':'S_bar', 'architecture':stage.value,
                        'state_dict':best_state, 'best_epoch':epoch, 'best_validation_loss':best,
                        'train_indices':train_idx, 'validation_indices':val_idx,
                        'directions':data.directions, 'weights':data.weights,
                        'data_path':str(Path(source_path).resolve()), 'options':vars(args),
                        'scale':scale.detach().cpu().numpy()}, directory/'model.pth')
        scheduler.step()
        if epoch == 1 or epoch % args.progress_interval == 0 or epoch == epochs:
            print(f'{stage.value} {epoch}/{epochs}: train={train_loss:.6g}, val={val_loss:.6g}, best={best:.6g}, seconds={time.monotonic()-start:.1f}', flush=True)
    model.load_state_dict(best_state)
    for k,v in frozen.items():
        if not torch.equal(model.state_dict()[k], v):
            raise RuntimeError('Frozen free energy changed during damage fitting')
    (directory/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    export(model, data, current, maximum, train_idx, val_idx, directory, history, args.batch_size)


def main(arguments=None):
    args = parser().parse_args(arguments)
    if min(args.free_energy_epochs,args.damage_epochs,args.batch_size,args.progress_interval) <= 0:
        raise ValueError('Epochs, batch size and progress interval must be positive')
    if not 0 < args.validation_fraction < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Invalid validation fraction/learning rate')
    if args.elastic_checkpoint and args.mode != 'damage':
        raise ValueError('--elastic-checkpoint is only supported with --mode damage')
    set_reproducible_seed(args.seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    elastic = load_sbar_data(args.no_damage_data)
    damage = load_sbar_data(args.damage_data) if args.mode != 'no_damage' else None
    if damage is not None:
        assert_sbar_pair(elastic, damage)
    train_idx, val_idx = train_test_split(np.arange(len(elastic.current)),
        test_size=args.validation_fraction, random_state=args.seed, stratify=elastic.case_index)
    architecture = Mode.FREE_ENERGY if args.mode == 'no_damage' else Mode.FREE_ENERGY_DAMAGE
    mean, std = float(elastic.current[train_idx].mean()), float(elastic.current[train_idx].std())
    model = ModularConstitutivePINN(elastic.directions, elastic.weights, mean, std,
                                   architecture=architecture).to(device)
    output = Path(args.output_dir)
    print(f'Device={device}; Xtrain=(2,{elastic.current.shape[1]},{len(elastic.current)}); current channel={elastic.current_channel}; maximum channel={elastic.maximum_channel}', flush=True)
    if args.elastic_checkpoint:
        checkpoint = torch.load(args.elastic_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'sbar_v1' or checkpoint.get('architecture') != 'free_energy':
            raise ValueError('Expected a no-damage S_bar checkpoint, not a Cauchy checkpoint')
        if Path(checkpoint['data_path']).resolve() != Path(args.no_damage_data).resolve():
            raise ValueError('Elastic checkpoint source path differs from no-damage data')
        for field in ('directions','weights'):
            if not np.array_equal(checkpoint[field], getattr(elastic,field)):
                raise ValueError('Checkpoint quadrature mismatch')
        train_idx, val_idx = checkpoint['train_indices'], checkpoint['validation_indices']
        if not np.array_equal(np.sort(np.r_[train_idx,val_idx]), np.arange(len(elastic.current))):
            raise ValueError('Checkpoint split does not match dataset')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        fit_stage(model, elastic, Mode.FREE_ENERGY, args, train_idx, val_idx,
                  output/'no_damage', args.no_damage_data)
    if damage is not None:
        fit_stage(model, damage, Mode.FREE_ENERGY_DAMAGE, args, train_idx, val_idx,
                  output/'damage', args.damage_data)


if __name__ == '__main__':
    main()
