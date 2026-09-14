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


if __name__ == '__main__':
    unittest.main()
