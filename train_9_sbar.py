"""Independently fit nine materials on proportional and prestretch histories."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from constitutive_pinn.config import ModelArchitecture as Mode
from constitutive_pinn.model import ModularConstitutivePINN
from constitutive_pinn.sbar_data import load_sbar_data
from constitutive_pinn.training import set_reproducible_seed, select_device
from train_sbar import fit_stage, export


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data')
    p.add_argument('--materials', nargs='+', type=int, choices=range(1, 10), default=list(range(1, 10)))
    p.add_argument('--output-dir', default='outputs/9_sbar_combined')
    p.add_argument('--damage-input', choices=['maximum_stretch', 'free_energy'],
                   default='maximum_stretch', help='Damage driver; free_energy uses current directional energies')
    p.add_argument('--training-data', choices=['combined', 'proportional'], default='combined',
                   help='Combined fits both loading families; proportional retains held-out prestretch testing')
    p.add_argument('--warmup-epochs', type=int, default=100,
                   help='Energy-only initialization on damaged targets; 0 disables warmup')
    p.add_argument('--warmup-learning-rate', type=float, default=None,
                   help='Defaults to --learning-rate')
    p.add_argument('--damage-epochs', type=int, default=1000)
    p.add_argument('--initial-damage', type=float, default=0.05,
                   help='Initial sigmoid damage output, not the linear bias')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--lr-step-size', type=int, default=200)
    p.add_argument('--lr-gamma', type=float, default=0.5)
    p.add_argument('--lr-scheduler', choices=['step', 'plateau'], default='plateau',
                   help='Joint-stage scheduler; warmup retains StepLR')
    p.add_argument('--lr-patience', type=int, default=40)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-threshold', type=float, default=1e-4,
                   help='Relative validation improvement required by plateau scheduler')
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=0.2)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--progress-interval', type=int, default=10)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--predict-only', action='store_true')
    return p


def audit(data, prestretch=False):
    expected_values = [1.01, 4, 6, 8] if prestretch else [0, 0.25, 0.5, 0.75, 1]
    values = data.prestretch if prestretch else data.ratios
    if values is None or not np.allclose(np.unique(values), expected_values):
        raise ValueError('Unexpected loading cases')
    reports = []
    for c in np.unique(data.case_index):
        ix = np.flatnonzero(data.case_index == c)
        x = data.lambda_x[ix]
        turns = np.r_[0, np.flatnonzero(np.diff(np.sign(np.diff(x))))+1, len(x)-1]
        if len(ix) != 4801 or not np.allclose(x[turns], [1,4,1,6,1,8,1,10,1]):
            raise ValueError(f'Unexpected x cycle in case {c}')
        reports.append({'case':int(c), 'samples':len(ix), 'loading_value':float(values[ix[0]]),
                        'lambda_x_turns':x[turns].tolist()})
    return {'samples':len(data.current), 'directions':data.current.shape[1],
            'current_channel_zero_based':data.current_channel,
            'maximum_channel_zero_based':data.maximum_channel, 'cases':reports}


def combine_data(proportional, prestretch):
    for field in ('directions', 'weights'):
        if not np.array_equal(getattr(proportional, field), getattr(prestretch, field)):
            raise ValueError('Cannot combine datasets with different quadrature')
    fields = {name: np.concatenate([getattr(proportional, name), getattr(prestretch, name)])
              for name in ('current', 'maximum', 'target', 'lambda_x', 'ratios')}
    # Cast before offsetting: original MATLAB case indices may be uint8.
    fields['case_index'] = np.r_[proportional.case_index.astype(np.int64),
        prestretch.case_index.astype(np.int64) + int(proportional.case_index.max())]
    fields['prestretch'] = np.r_[np.full(len(proportional.current), np.nan), prestretch.prestretch]
    return replace(proportional, **fields)


def local_indices(indices, start, count):
    indices = np.asarray(indices)
    return indices[(indices >= start) & (indices < start + count)] - start


def fit_material(model, data, args, train_idx, val_idx, output, source_path):
    if args.warmup_epochs:
        print(f'Energy warmup: {args.warmup_epochs} epochs on selected damaged targets', flush=True)
        fit_stage(model, data, Mode.FREE_ENERGY, args, train_idx, val_idx,
                  output/'warmup', source_path, epochs=args.warmup_epochs,
                  learning_rate=args.warmup_learning_rate, strategy='damaged_energy_warmup')
    # fit_stage restores the best warmup weights. Reuse this model, then enable
    # both branches with a fresh optimizer and scheduler; do not reinitialize.
    print(f'Joint energy/damage training: {args.damage_epochs} epochs', flush=True)
    fit_stage(model, data, Mode.FREE_ENERGY_DAMAGE, args, train_idx, val_idx,
              output/('combined' if getattr(args, 'training_data', 'proportional') == 'combined' else 'proportional'), source_path, joint=True,
              strategy='warmup_joint' if args.warmup_epochs else 'joint')


def main(arguments=None, *, defaults=None):
    argument_parser = parser()
    if defaults:
        argument_parser.set_defaults(**defaults)
    args = argument_parser.parse_args(arguments)
    if min(args.damage_epochs, args.batch_size, args.progress_interval) <= 0:
        raise ValueError('Epochs, batch size and progress interval must be positive')
    if not 0 < args.validation_fraction < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Invalid validation fraction/learning rate')
    if args.warmup_epochs < 0 or args.lr_step_size <= 0 or not 0 < args.lr_gamma <= 1:
        raise ValueError('Invalid warmup epochs or scheduler settings')
    if not np.isfinite(args.initial_damage) or not 0 < args.initial_damage < 1:
        raise ValueError('Initial damage must be finite and strictly between 0 and 1')
    if (args.lr_patience < 0 or not 0 < args.lr_factor < 1
            or not 0 <= args.lr_threshold < 1
            or not np.isfinite(args.min_lr) or not 0 < args.min_lr <= args.learning_rate):
        raise ValueError('Invalid plateau scheduler settings')
    if args.warmup_learning_rate is not None and (not np.isfinite(args.warmup_learning_rate) or args.warmup_learning_rate <= 0):
        raise ValueError('Warmup learning rate must be finite and positive')
    if args.audit_only and args.predict_only:
        raise ValueError('Choose audit-only or predict-only')
    torch.set_num_threads(4)
    device = select_device(args.device)
    summaries = {}
    for material in dict.fromkeys(args.materials):
        source = Path(args.data_root)/str(material)
        train_path = source/'proportional_biaxial_Sbar.mat'
        train = load_sbar_data(train_path)
        test = load_sbar_data(source/'orthogonal_prestretch_x_cyclic_Sbar.mat')
        proportional = train
        combined = args.training_data == 'combined'
        fit_name = 'combined' if combined else 'proportional'
        prestretch_name = 'prestretch_fit' if combined else 'prestretch_test'
        for field in ('directions', 'weights'):
            if not np.array_equal(getattr(train, field), getattr(test, field)):
                raise ValueError(f'Material {material}: quadrature mismatch')
        report = {'material':material, 'deformation_coupling':((material-1)%3)*0.5,
                  'damage_coupling':((material-1)//3)*0.5,
                  'proportional':audit(train), 'prestretch':audit(test, True)}
        report['training_data'] = args.training_data
        report['damage_input'] = args.damage_input
        report['source_paths'] = [str(train_path.resolve()),
            str((source/'orthogonal_prestretch_x_cyclic_Sbar.mat').resolve())]
        if combined:
            train = combine_data(proportional, test)
        report['fit_samples'] = len(train.current)
        output = Path(args.output_dir)/str(material)
        output.mkdir(parents=True, exist_ok=True)
        (output/'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Material {material}: proportional {len(proportional.current)}, prestretch {len(test.current)}; training_data={args.training_data}, fit samples={len(train.current)}; audit passed', flush=True)
        if args.audit_only:
            continue
        # Start each material from the same reproducible initialization, never
        # transfer learned weights or input statistics between materials.
        set_reproducible_seed(args.seed)
        train_idx, val_idx = train_test_split(np.arange(len(train.current)),
            test_size=args.validation_fraction, random_state=args.seed, stratify=train.case_index)
        model = ModularConstitutivePINN(train.directions, train.weights,
            float(train.current[train_idx].mean()), float(train.current[train_idx].std()),
            architecture=Mode.FREE_ENERGY_DAMAGE, initial_damage=args.initial_damage,
            damage_input=args.damage_input).to(device)
        if args.predict_only:
            ckpt = torch.load(output/fit_name/'model.pth', map_location='cpu', weights_only=False)
            # Checkpoint metadata is authoritative on replay; missing means legacy.
            model.damage_input = ckpt.get('damage_input', 'maximum_stretch')
            if model.damage_input not in ('maximum_stretch', 'free_energy'):
                raise ValueError('Unknown checkpoint damage input')
            report['damage_input'] = model.damage_input
            (output/'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            if (ckpt.get('format') != 'sbar_v1' or ckpt.get('stress_measure') != 'S_bar'
                    or ckpt.get('training_strategy') not in ('joint', 'warmup_joint')
                    or ckpt.get('architecture') != Mode.FREE_ENERGY_DAMAGE.value
                    or Path(ckpt['data_path']).resolve() != train_path.resolve()
                    or ckpt.get('options', {}).get('training_data', 'proportional') != args.training_data):
                raise ValueError('Expected this material\'s joint S_bar checkpoint')
            for field in ('directions', 'weights'):
                if not np.array_equal(ckpt[field], getattr(train, field)):
                    raise ValueError('Checkpoint quadrature mismatch')
            train_idx, val_idx = ckpt['train_indices'], ckpt['validation_indices']
            if not np.array_equal(np.sort(np.r_[train_idx,val_idx]), np.arange(len(train.current))):
                raise ValueError('Checkpoint split mismatch')
            model.load_state_dict(ckpt['state_dict'])
            history = json.loads((output/fit_name/'history.json').read_text(encoding='utf-8'))
            export(model, train, torch.as_tensor(train.current, device=device),
                   torch.as_tensor(train.maximum, device=device), train_idx, val_idx,
                   output/fit_name, history, args.batch_size)
        else:
            fit_material(model, train, args, train_idx, val_idx, output, train_path)
        if combined:
            proportional_dir = output/'proportional'
            proportional_dir.mkdir(exist_ok=True)
            n = len(proportional.current)
            export(model, proportional, torch.as_tensor(proportional.current, device=device),
                   torch.as_tensor(proportional.maximum, device=device),
                   local_indices(train_idx, 0, n), local_indices(val_idx, 0, n),
                   proportional_dir, [], args.batch_size)
        test_dir = output/prestretch_name
        test_dir.mkdir(exist_ok=True)
        empty = np.array([], dtype=int)
        export(model, test, torch.as_tensor(test.current, device=device),
               torch.as_tensor(test.maximum, device=device),
               local_indices(train_idx, len(proportional.current), len(test.current)) if combined else empty,
               local_indices(val_idx, len(proportional.current), len(test.current)) if combined else empty,
               test_dir, [], args.batch_size)
        summaries[str(material)] = {
            'proportional':json.loads((output/'proportional'/'metrics.json').read_text()),
            prestretch_name:json.loads((test_dir/'metrics.json').read_text())}
        if combined:
            summaries[str(material)]['combined'] = json.loads((output/'combined'/'metrics.json').read_text())
    if summaries:
        # Material-specific files remain authoritative when running a subset.
        name = 'summary_' + '_'.join(summaries) + '.json'
        (Path(args.output_dir)/name).write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
