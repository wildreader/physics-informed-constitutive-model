"""Serial driver, chain rule, training gradients, and checkpoint compatibility."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from constitutive_pinn.model import ModularConstitutivePINN
from constitutive_pinn.config import ModelArchitecture as Mode
from constitutive_pinn.checkpoint import save_checkpoint, load_checkpoint, restore_checkpoint


class Quadratic(torch.nn.Module):
    def forward(self, x, weights):
        return x.square() / 2


class EnergyDamage(torch.nn.Module):
    def forward(self, phi, weights):
        return torch.sigmoid(phi)


def model(driver='free_energy', directions=3):
    dirs = np.tile(np.eye(3), (37, 1))[:directions]
    return ModularConstitutivePINN(dirs, np.ones(directions)/directions, 0., 1.,
        architecture=Mode.FREE_ENERGY_DAMAGE, initial_damage=0.06, damage_input=driver)


class SerialDamageTests(unittest.TestCase):
    def test_serial_chain_rule_and_history_independence(self):
        m = model().eval()
        m.free_energy_module = Quadratic()
        m.damage_module = EnergyDamage()
        x = torch.tensor([[1., 1.5, 2.]])
        phi = x.square()/2
        d = torch.sigmoid(phi)
        expected = torch.cat(((1-d-phi*d*(1-d))/3, torch.zeros(1, 3)), dim=1)
        with torch.no_grad():
            actual = m.forward_sbar(x)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(m.forward_sbar(x, x*9), actual)

    def test_110_raw_energies_and_joint_gradients(self):
        m = model(directions=110)
        # Nonzero last weights exercise gradients through the whole damage branch.
        with torch.no_grad():
            m.damage_module.predictor[-2].weight.fill_(0.01)
        x = torch.linspace(1, 3, 220).reshape(2, 110)
        expected = m.free_energy_module(x/(1+1e-8), m.norm_weights)
        with patch.object(m.damage_module, 'forward', wraps=m.damage_module.forward) as call:
            output = m.forward_sbar(x)
        torch.testing.assert_close(call.call_args.args[0], expected)
        self.assertEqual(call.call_args.args[0].shape, (2, 110))
        output.square().sum().backward()
        for branch in (m.free_energy_module, m.damage_module):
            grads = [p.grad for p in branch.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.assertTrue(all(p.grad is None for p in m.volumetric_module.parameters()))

    def test_initial_damage_and_legacy_state_compatibility(self):
        old, serial = model('maximum_stretch'), model()
        serial.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(old.parameter_summary(), serial.parameter_summary())
        x = torch.tensor([[1., 2., 3.]])
        serial.eval()
        serial.activate_modules(Mode.FREE_ENERGY)
        elastic = serial.forward_sbar(x)
        serial.activate_modules(Mode.FREE_ENERGY_DAMAGE)
        torch.testing.assert_close(serial.forward_sbar(x), elastic*0.94)
        old.eval()
        torch.testing.assert_close(old.forward_sbar(x, x), serial.forward_sbar(x))
        with self.assertRaises(ValueError):
            old.forward_sbar(x)

    def test_checkpoint_roundtrip_and_missing_metadata(self):
        m = model().eval()
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pth'
            save_checkpoint(path, m, Mode.FREE_ENERGY_DAMAGE, 0., 1., 1e-4, [1.]*6)
            restored = model('maximum_stretch').eval()
            restore_checkpoint(restored, load_checkpoint(path, 'cpu'))
            self.assertEqual(restored.damage_input, 'free_energy')
            x = torch.ones(1, 3)
            torch.testing.assert_close(restored.forward_sbar(x), m.forward_sbar(x))
            raw = torch.load(path, weights_only=True)
            del raw['damage_input']
            torch.save(raw, path)
            restore_checkpoint(restored, load_checkpoint(path, 'cpu'))
            self.assertEqual(restored.damage_input, 'maximum_stretch')

    def test_serial_entry_defaults(self):
        from train_9_sbar_serial import main, DEFAULTS
        from pred_9_sbar_serial import main as predict
        self.assertEqual(DEFAULTS['initial_damage'], 0.06)
        self.assertEqual(DEFAULTS['training_data'], 'proportional')
        with patch('train_9_sbar_serial.run') as run:
            main(['--materials', '1'])
            self.assertEqual(run.call_args.kwargs['defaults'], DEFAULTS)
        with patch('pred_9_sbar_serial.train_main') as run:
            predict(['--materials', '1'])
            self.assertEqual(run.call_args.args[0], ['--materials', '1', '--predict-only'])


if __name__ == '__main__':
    unittest.main()
