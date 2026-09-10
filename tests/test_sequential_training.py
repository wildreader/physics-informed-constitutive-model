"""验证物理开关、真实自动微分、最佳权重继承与单检查点保存。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.io
import torch

from constitutive_pinn import ModelArchitecture as Mode, ModularConstitutivePINN
from constitutive_pinn.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from constitutive_pinn.data import (
    TrainingData, assert_compatible_stages, default_training_data_path,
    default_prediction_data_path, load_training_data, load_prediction_data,
)
from constitutive_pinn.training import TrainingOptions, build_training_loaders, fit_model
import constitutive_pinn.training as training


torch.set_num_threads(1)
DEVICE = torch.device("cpu")


def fixture(mode):
    directions = np.concatenate((np.eye(3), -np.eye(3))).astype(np.float32)
    weights = np.full(6, 1 / 6, dtype=np.float32)
    stretch = np.linspace(1.1, 2.0, 12)
    f_bar = np.zeros((12, 3, 3))
    f_bar[:, 0, 0] = stretch
    f_bar[:, 1, 1] = f_bar[:, 2, 2] = stretch ** -0.5
    current = np.linalg.norm(np.einsum("bij,dj->bdi", f_bar, directions), axis=-1)
    maximum = np.maximum(current, 1)
    j = (1 + 1e-4 * (stretch - 1)) if mode.use_volumetric else np.ones(12)
    f = f_bar * j[:, None, None] ** (1 / 3)
    c = np.einsum("bji,bjk->bik", f, f)
    columns = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
    force = np.maximum(current - 1, 0) ** 3
    if mode.use_damage:
        survival = np.maximum((12 - maximum) / 12, 0)
        force *= 0.5 * survival + 0.5 * (survival @ weights)[:, None]
    s_bar = np.einsum("bd,di,dj->bij", weights * force / current, directions, directions)
    cinv = np.linalg.inv(c)
    s_iso = (s_bar - np.einsum("bij,bij->b", c, s_bar)[:, None, None] * cinv / 3)
    s_iso *= j[:, None, None] ** (-2 / 3)
    sigma = f @ s_iso @ f.transpose(0, 2, 1) / j[:, None, None]
    if mode.use_volumetric:
        sigma += (1e6 * (j - 1))[:, None, None] * np.eye(3)
    return TrainingData(
        lambda_current=current.astype(np.float32),
        lambda_maximum=maximum.astype(np.float32), j_ratio=j[:, None],
        c_vector=c[:, columns[0], columns[1]].astype(np.float32),
        f_vector=f.reshape(-1, 9).astype(np.float32),
        sigma_total=sigma[:, columns[0], columns[1]].astype(np.float32),
        directions=directions, weights=weights, case_lengths=np.array([12]),
        stretch_path=stretch, stretch_path_x=stretch, pre_stretch_length=1,
        delta_j_ref=1e-4, model_architecture=mode.value,
    )


def make_model(mode):
    torch.manual_seed(42)
    data = fixture(Mode.FREE_ENERGY)
    return ModularConstitutivePINN(data.directions, data.weights, *data.normalization,
                                   architecture=mode)


def inputs(data):
    return tuple(torch.from_numpy(getattr(data, name)) for name in (
        "lambda_current", "lambda_maximum", "j_ratio", "c_vector", "f_vector"
    ))


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


class SequentialTests(unittest.TestCase):
    def test_all_modes_have_identical_three_modules_and_initialization(self):
        models = [make_model(mode) for mode in Mode]
        for model in models:
            self.assertEqual(set(model.state_dict()), set(models[0].state_dict()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, models[0].state_dict()[key]), key)
            d = model.damage_module(torch.zeros(2, 6), model.norm_weights)
            self.assertTrue(torch.equal(d, torch.full_like(d, 0.5)))
            self.assertTrue(torch.equal(model.volumetric_module(torch.ones(2, 1)),
                                        torch.zeros(2, 1)))

    def test_disabled_modules_not_called_and_switch_does_not_reinitialize(self):
        model = make_model(Mode.FULL).eval()
        original = snapshot(model)
        model.activate_modules(Mode.FREE_ENERGY)
        x = list(inputs(fixture(Mode.FREE_ENERGY)))
        x[1] = None
        with patch.object(model.damage_module, "forward", side_effect=AssertionError), \
             patch.object(model.volumetric_module, "forward", side_effect=AssertionError), \
             torch.no_grad():
            total, iso, vol, p = model(*x, return_components=True)
        torch.testing.assert_close(total, iso, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(vol).item(), 0)
        self.assertEqual(torch.count_nonzero(p).item(), 0)
        model.activate_modules(Mode.FREE_ENERGY_DAMAGE)
        with patch.object(model.volumetric_module, "forward", side_effect=AssertionError):
            _, _, vol, p = model(*inputs(fixture(Mode.FREE_ENERGY_DAMAGE)), return_components=True)
        self.assertEqual(torch.count_nonzero(vol).item() + torch.count_nonzero(p).item(), 0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_real_training_freezes_old_modules_and_restores_best_not_last(self):
        options = TrainingOptions(batch_size=6, free_energy_epochs=3, damage_epochs=3,
                                  volumetric_epochs=3, scheduler_step_size=1)
        model = make_model(Mode.FULL)
        loaders = {mode: build_training_loaders(fixture(mode), options) for mode in Mode}
        real_epoch = training._run_epoch
        counts = {mode: 0 for mode in Mode}
        best_states = {}
        observed_lr = []

        def run_epoch(net, loader, scale, device, optimizer):
            mode = net.active_architecture
            before = snapshot(net)
            if optimizer is not None:
                if counts[mode] == 0:
                    # 新阶段开始时必须继承前面最优，而非最后 epoch 的参数。
                    for prior in mode.training_stages[:-1]:
                        for key, value in before.items():
                            if key.startswith(prior.module_name + "_module."):
                                self.assertTrue(torch.equal(value, best_states[prior][key]))
                observed_lr.append(optimizer.param_groups[0]["lr"])
            loss = real_epoch(net, loader, scale, device, optimizer)
            self.assertTrue(np.isfinite(loss))
            if optimizer is not None:
                changed = []
                for key, value in net.state_dict().items():
                    if not torch.equal(value, before[key]):
                        changed.append(key)
                self.assertTrue(changed, f"{mode.value} should receive real gradients")
                self.assertTrue(all(k.startswith(mode.module_name + "_module.") for k in changed))
                return loss
            counts[mode] += 1
            if counts[mode] == 2:
                best_states[mode] = snapshot(net)
            # 人为构造第 2 轮最优，第 3 轮退化；训练更新仍使用真实 autograd。
            return [2.0, 1.0, 3.0][counts[mode] - 1]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            saves = []

            def save_best(info):
                saves.append(info)
                save_checkpoint(path, model, Mode.FULL, *fixture(Mode.FREE_ENERGY).normalization,
                                1e-4, loaders[Mode.FULL].sigma_scale.tolist(), training=info)

            with patch.object(training, "_run_epoch", side_effect=run_epoch):
                history = fit_model(model, loaders, options, DEVICE, on_final_best=save_best)
            self.assertEqual([s["best_epoch"] for s in history.stages], [2, 2, 2])
            self.assertEqual(len(saves), 2)  # 只在最终阶段前两轮改善时保存。
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(history.stage_names, ["free_energy"] * 3 + ["damage"] * 3 + ["volumetric"] * 3)
            np.testing.assert_allclose(observed_lr, [0.002, 0.001, 0.0005,
                                                    0.001, 0.0005, 0.00025,
                                                    0.001, 0.0005, 0.00025])
            checkpoint = load_checkpoint(path, DEVICE, Mode.FULL)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, best_states[Mode.FULL][key]), key)
                self.assertTrue(torch.equal(value, checkpoint.state_dict[key]), key)
            restored = make_model(Mode.FULL).eval()
            restore_checkpoint(restored, checkpoint)
            with torch.no_grad():
                torch.testing.assert_close(model(*inputs(fixture(Mode.FULL))),
                                           restored(*inputs(fixture(Mode.FULL))))

    def test_single_and_two_stage_plans_and_training_loss_selection(self):
        options = TrainingOptions(batch_size=6, free_energy_epochs=1, damage_epochs=1,
                                  volumetric_epochs=1, selection_loss="train")
        for mode in (Mode.FREE_ENERGY, Mode.FREE_ENERGY_DAMAGE):
            with self.subTest(mode=mode):
                model = make_model(mode)
                loaders = {s: build_training_loaders(fixture(s), options) for s in mode.training_stages}
                history = fit_model(model, loaders, options, DEVICE)
                self.assertEqual(len(history.stages), len(mode.training_stages))
                for index, stage in enumerate(history.stages):
                    self.assertEqual(stage["best_loss"], history.train_loss[index])
                self.assertIs(model.active_architecture, mode)

    def test_reject_missing_or_wrong_stage_data_and_zero_active_epochs(self):
        options = TrainingOptions()
        model = make_model(Mode.FULL)
        with self.assertRaises(ValueError):
            fit_model(model, {}, options, DEVICE)
        loaders = {s: build_training_loaders(fixture(s), options) for s in Mode}
        loaders[Mode.FREE_ENERGY] = loaders[Mode.FULL]
        with self.assertRaises(ValueError):
            fit_model(model, loaders, options, DEVICE)
        with self.assertRaises(ValueError):
            replace(options, damage_epochs=0).validate(Mode.FULL)
        with self.assertRaises(ValueError):
            assert_compatible_stages(fixture(Mode.FREE_ENERGY),
                                     replace(fixture(Mode.FULL), weights=np.ones(6)))

    def test_legacy_checkpoint_migration_and_missing_active_weights_rejected(self):
        reverse = {"free_energy_module.extractor.": "elastic_extractor.",
                   "free_energy_module.predictor.": "elastic_predictor.",
                   "damage_module.extractor.": "damage_extractor.",
                   "damage_module.predictor.": "damage_predictor.",
                   "volumetric_module.": "volumetric_net."}
        for mode in Mode:
            model = make_model(mode).eval()
            legacy = {}
            for key, value in model.state_dict().items():
                if key.startswith("damage_module.") and not mode.use_damage:
                    continue
                if key.startswith("volumetric_module.") and not mode.use_volumetric:
                    continue
                for new, old in reverse.items():
                    if key.startswith(new):
                        key = old + key[len(new):]
                        break
                legacy[key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "legacy.pth"
                torch.save({"format_version": 1, "architecture": mode.value,
                            "model_state_dict": legacy}, path)
                checkpoint = load_checkpoint(path, DEVICE, mode)
                restored = make_model(mode).eval()
                restore_checkpoint(restored, checkpoint)
                torch.testing.assert_close(model(*inputs(fixture(mode))),
                                           restored(*inputs(fixture(mode))))
                broken = dict(checkpoint.state_dict)
                del broken["elastic_extractor.0.weight"]
                with self.assertRaises(RuntimeError):
                    restore_checkpoint(restored, replace(checkpoint, state_dict=broken))

    def test_new_checkpoint_roundtrip_all_modes_and_corrupt_switches_rejected(self):
        for mode in Mode:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = make_model(mode).eval()
                path = Path(directory) / "model.pth"
                save_checkpoint(path, model, mode, *fixture(mode).normalization,
                                1e-4, [1.0] * 6)
                checkpoint = load_checkpoint(path, DEVICE, mode)
                restored = make_model(mode).eval()
                restore_checkpoint(restored, checkpoint)
                self.assertEqual(set(checkpoint.state_dict), set(make_model(Mode.FULL).state_dict()))
                with torch.no_grad():
                    torch.testing.assert_close(model(*inputs(fixture(mode))),
                                               restored(*inputs(fixture(mode))))
                raw = torch.load(path, weights_only=True)
                raw["enabled_modules"]["damage"] = not mode.use_damage
                torch.save(raw, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path, DEVICE, mode)


class MatlabDataTests(unittest.TestCase):
    @unittest.skipUnless(default_training_data_path(Mode.FULL).exists(), "先生成 MATLAB 数据")
    def test_all_six_datasets_match_analytic_module_formulas(self):
        for mode in Mode:
            reference = load_training_data(default_training_data_path(mode), mode)
            for kind in ("train", "test"):
                path = (default_training_data_path(mode) if kind == "train"
                        else default_prediction_data_path(mode))
                data = reference if kind == "train" else load_prediction_data(path, mode)
                raw = scipy.io.loadmat(path)
                self.assertEqual(int(raw["dataset_format_version"].item()), 2)
                suffix = "export" if kind == "train" else "test"
                pressure = raw[f"Y_p_{suffix}"]
                if not mode.use_volumetric:
                    np.testing.assert_array_equal(data.j_ratio, 1)
                    np.testing.assert_array_equal(pressure, 0)
                    self.assertNotIn("k_vol", raw)
                else:
                    np.testing.assert_allclose(pressure, 1e6 * (data.j_ratio - 1), atol=1e-9)
                if not mode.use_damage:
                    np.testing.assert_array_equal(data.lambda_maximum, 1)
                indices = np.linspace(0, data.lambda_current.shape[0] - 1, 35, dtype=int)
                # 使用原始 double MAT 数据独立重算，避免高伸长下 float32 放大误差。
                ls = raw[f"X_La_curr_{suffix}"][indices]
                ls_max = raw[f"X_La_max_{suffix}"][indices]
                weights = reference.weights.astype(np.float64)
                directions = reference.directions.astype(np.float64)
                force = np.maximum(ls - 1, 0) ** 3
                if mode.use_damage:
                    local = np.maximum((12 - np.maximum(ls_max, 1)) / 12, 0)
                    force *= 0.5 * local + 0.5 * (local @ weights)[:, None]
                sbar = np.einsum("bd,di,dj->bij", weights * force / ls, directions, directions)
                f = raw[f"X_F_{suffix}"][indices].reshape(-1, 3, 3)
                c = f.transpose(0, 2, 1) @ f
                j = data.j_ratio[indices, 0]
                s = (sbar - np.einsum("bij,bij->b", c, sbar)[:, None, None] * np.linalg.inv(c) / 3)
                s *= j[:, None, None] ** (-2 / 3)
                sigma = f @ s @ f.transpose(0, 2, 1) / j[:, None, None]
                if mode.use_volumetric:
                    sigma += pressure[indices, :, None] * np.eye(3)
                vec = sigma[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
                np.testing.assert_allclose(vec, data.sigma_total[indices], rtol=2e-5, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
