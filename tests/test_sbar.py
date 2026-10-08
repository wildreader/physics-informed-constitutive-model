"""S_bar endpoint must bypass projection and retain energy/damage gradients."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from constitutive_pinn.model import ModularConstitutivePINN
from constitutive_pinn.config import ModelArchitecture as Mode
from constitutive_pinn.sbar_data import load_sbar_data


class QuadraticEnergy(torch.nn.Module):
    def forward(self, values, weights):
        return values.square() / 2


class SbarTests(unittest.TestCase):
    def test_initial_damage_preserves_95_percent_of_warmup_stress(self):
        model = ModularConstitutivePINN(np.eye(3), np.ones(3)/3, 0., 1.,
            architecture=Mode.FREE_ENERGY_DAMAGE, initial_damage=0.05)
        current = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
        torch.testing.assert_close(model.damage_module(current, model.norm_weights),
                                   torch.full_like(current, 0.05))
        model.eval()
        model.activate_modules(Mode.FREE_ENERGY)
        elastic = model.forward_sbar(current)
        model.activate_modules(Mode.FREE_ENERGY_DAMAGE)
        torch.testing.assert_close(model.forward_sbar(current, current), 0.95 * elastic)

    def test_plateau_uses_validation_and_respects_minimum_lr(self):
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from constitutive_pinn.sbar_data import SbarData
        from train_sbar import fit_stage
        from train_9_sbar import parser
        args = parser().parse_args(['--damage-epochs', '4', '--lr-patience', '0',
                                   '--min-lr', '0.0003'])
        model = ModularConstitutivePINN(np.eye(3), np.ones(3)/3, 0., 1.,
            architecture=Mode.FREE_ENERGY_DAMAGE, initial_damage=args.initial_damage)
        current = np.array([[1,1,1], [2,1,0.5], [3,1,1/3], [2,1,0.5]], dtype=np.float32)
        data = SbarData(current, np.maximum.accumulate(current, axis=0),
                        np.ones((4,6), dtype=np.float32), np.eye(3), np.ones(3)/3,
                        np.ones(4), current[:,0], np.zeros(4), 0, 1)
        original_forward = model.forward_sbar
        def fixed_validation(x, maximum):
            if model.training:
                return original_forward(x, maximum)
            return torch.full((len(x), 6), 2.0)
        with TemporaryDirectory() as tmp, patch('train_sbar.export'), patch.object(
                model, 'forward_sbar', side_effect=fixed_validation):
            fit_stage(model, data, Mode.FREE_ENERGY_DAMAGE, args, np.array([0,1,2]),
                      np.array([3]), Path(tmp), 'synthetic.mat', joint=True)
            history = json.loads((Path(tmp)/'history.json').read_text())
        self.assertEqual([h['validation_loss'] for h in history], [1.0]*4)
        np.testing.assert_allclose([h['learning_rate'] for h in history],
                                   [0.001, 0.001, 0.0005, 0.0003])
        self.assertEqual(history[-1]['next_learning_rate'], 0.0003)

    def test_analytic_integral_without_projection(self):
        directions = np.eye(3, dtype=np.float32)
        model = ModularConstitutivePINN(directions, np.ones(3)/3, 0., 1., architecture=Mode.FREE_ENERGY)
        model.free_energy_module = QuadraticEnergy()
        model.eval()
        current = torch.tensor([[1., 2., 3.]])
        with patch.object(model.projection_layer, 'forward', side_effect=AssertionError('projection executed')):
            with torch.no_grad():
                result = model.forward_sbar(current)
        torch.testing.assert_close(result, torch.tensor([[1/3,1/3,1/3,0.,0.,0.]]))

    def test_damage_backward_with_frozen_energy(self):
        model = ModularConstitutivePINN(np.eye(3), np.ones(3)/3, 0., 1., architecture=Mode.FREE_ENERGY_DAMAGE)
        model.set_training_stage(Mode.FREE_ENERGY_DAMAGE)
        current = torch.tensor([[1.,2.,3.]], requires_grad=True)
        maximum = current * 2  # history must be held fixed even if caller connects it
        result = model.forward_sbar(current, maximum)
        result.square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.damage_module.parameters()))
        self.assertTrue(all(p.grad is None for p in model.free_energy_module.parameters()))

    def test_actual_data_history_and_layout(self):
        data = load_sbar_data('proportional_biaxial_Sbar.mat')
        self.assertEqual((data.current_channel,data.maximum_channel),(0,1))
        self.assertEqual(data.current.shape,(24005,110))
        self.assertEqual(data.target.shape,(24005,6))

    def test_full_rejects_sbar(self):
        model = ModularConstitutivePINN(np.eye(3), np.ones(3)/3, 0., 1., architecture=Mode.FULL)
        with self.assertRaises(ValueError):
            model.forward_sbar(torch.ones(1,3))

    def test_joint_fit_updates_energy_and_damage_only(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from types import SimpleNamespace
        from constitutive_pinn.sbar_data import SbarData
        from train_sbar import fit_stage
        model = ModularConstitutivePINN(np.eye(3), np.ones(3)/3, 0., 1.,
                                        architecture=Mode.FREE_ENERGY_DAMAGE)
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        current = np.array([[1,1,1], [2,1,0.5], [3,1,1/3], [2,1,0.5]], dtype=np.float32)
        data = SbarData(current, np.maximum.accumulate(current, axis=0),
                        np.ones((4,6), dtype=np.float32), np.eye(3), np.ones(3)/3,
                        np.ones(4), current[:,0], np.zeros(4), 0, 1)
        args = SimpleNamespace(learning_rate=0.001, damage_epochs=2, batch_size=2, progress_interval=1,
                               warmup_epochs=2, warmup_learning_rate=0.002, lr_step_size=1, lr_gamma=0.5)
        from train_9_sbar import fit_material
        stages = []
        def checked_fit(*pos, **kw):
            if kw.get('joint'):
                saved = torch.load(pos[6].parent/'warmup'/'model.pth', weights_only=False)
                for k,v in model.state_dict().items():
                    torch.testing.assert_close(v.cpu(), saved['state_dict'][k])
                self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items()
                                    if k.startswith(('damage_module.', 'volumetric_module.'))))
            stages.append(pos[2])
            return fit_stage(*pos, **kw)
        with TemporaryDirectory() as tmp, patch('train_sbar.export'):
            with patch('train_9_sbar.fit_stage', side_effect=checked_fit):
                fit_material(model, data, args, np.array([0,1,2]), np.array([3]), Path(tmp), 'synthetic.mat')
            self.assertEqual(stages, [Mode.FREE_ENERGY, Mode.FREE_ENERGY_DAMAGE])
            saved = torch.load(Path(tmp)/'proportional'/'model.pth', weights_only=False)
            self.assertEqual(saved['training_strategy'], 'warmup_joint')
        final = model.state_dict()
        for prefix in ('free_energy_module.', 'damage_module.'):
            self.assertTrue(any(not torch.equal(v, final[k]) for k,v in initial.items() if k.startswith(prefix)))
        self.assertTrue(all(torch.equal(v, final[k]) for k,v in initial.items() if k.startswith('volumetric_module.')))

    def test_zero_warmup_keeps_direct_joint_path(self):
        from train_9_sbar import parser, fit_material
        from pathlib import Path
        args = parser().parse_args(['--warmup-epochs', '0'])
        with patch('train_9_sbar.fit_stage') as fit:
            fit_material(None, None, args, None, None, Path('unused'), 'unused.mat')
        fit.assert_called_once()
        self.assertTrue(fit.call_args.kwargs['joint'])
        self.assertEqual(fit.call_args.kwargs['strategy'], 'joint')

    def test_prestretch_history_is_retained_and_checked(self):
        from scipy.io import loadmat
        path = 'data/1/orthogonal_prestretch_x_cyclic_Sbar.mat'
        data = load_sbar_data(path)
        self.assertEqual(data.current.shape, (19204,110))
        np.testing.assert_allclose(np.unique(data.prestretch), [1.01,4,6,8])
        self.assertGreater(data.maximum[4801].max(), data.current[4801].max())
        raw = loadmat(path, simplify_cells=True)
        raw['Xtrain'][1,:,4801] = 1
        with patch('constitutive_pinn.sbar_data.loadmat', return_value=raw):
            with self.assertRaises(ValueError):
                load_sbar_data(path)


if __name__ == '__main__':
    unittest.main()
