import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock

import torch

from glyph_sr.checkpoints import load_state_dict, load_ts_checkpoint


def tiny_model():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.diffusion_model = torch.nn.Module()
    model.model.diffusion_model.project_modules = torch.nn.Linear(2, 2)
    model.model.diffusion_model.base = torch.nn.Linear(2, 2)
    return model


class CheckpointTests(unittest.TestCase):
    def test_old_state_tensors_are_released_before_device_transfer(self):
        for kind in ("full", "lightning", "inner", "prefixed_inner"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                source, target = tiny_model(), tiny_model()
                for parameter in source.parameters():
                    torch.nn.init.constant_(parameter, 3)
                if kind == "prefixed_inner":
                    target.register_buffer("metric", torch.tensor([7.0]))
                snapshots = []

                def record_snapshots(module):
                    original = module.state_dict

                    def state_dict(*args, **kwargs):
                        result = original(*args, **kwargs)
                        snapshots.extend(weakref.ref(value) for value in result.values())
                        return result

                    module.state_dict = state_dict

                # Regular functions avoid mocks retaining recursive destination dictionaries.
                record_snapshots(target)
                record_snapshots(target.model)
                original_to = target.to

                def move_to(device):
                    self.assertTrue(snapshots)
                    self.assertTrue(all(reference() is None for reference in snapshots))
                    report = target._ts_checkpoint_load_report
                    self.assertEqual(
                        report["matched_tensor_count"], report["expected_tensor_count"]
                    )
                    self.assertEqual(report["missing_key_count"], 0)
                    return original_to(device)

                target.to = Mock(side_effect=move_to)
                weights = source.model.state_dict() if kind == "inner" else source.state_dict()
                payload = {"state_dict": weights, "epoch": 1} if kind == "lightning" else weights
                path = Path(folder) / "weights.ckpt"
                torch.save(payload, path)
                self.assertIs(load_ts_checkpoint(target, path, "cpu"), target)
                target.to.assert_called_once_with("cpu")
                self.assertTrue(all(torch.all(parameter == 3) for parameter in target.parameters()))

    def test_supported_containers_restore_complete_weights(self):
        for kind in ("lightning", "full", "inner"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                source, target = tiny_model(), tiny_model()
                for value in source.parameters():
                    torch.nn.init.constant_(value, 3)
                state = source.model.state_dict() if kind == "inner" else source.state_dict()
                payload = {"state_dict": state, "epoch": 4} if kind == "lightning" else state
                path = Path(folder) / "weights.ckpt"
                torch.save(payload, path)
                self.assertIs(load_ts_checkpoint(target, path, "cpu"), target)
                for key, value in source.state_dict().items():
                    torch.testing.assert_close(target.state_dict()[key], value)
                report = target._ts_checkpoint_load_report
                self.assertEqual(report["matched_tensor_count"], report["expected_tensor_count"])
                self.assertEqual(report["matched_projection_tensor_count"], 2)
                self.assertEqual(report["missing_key_count"], 0)

    def test_prefixed_inner_weights_can_omit_unrelated_outer_metrics(self):
        source, target = tiny_model(), tiny_model()
        target.register_buffer("metric", torch.tensor([7.0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.ckpt"
            torch.save(source.state_dict(), path)
            load_ts_checkpoint(target, path, "cpu")
        self.assertEqual(target.metric.item(), 7)
        self.assertEqual(
            target._ts_checkpoint_load_report["load_target"],
            "inner_model_stripped_model_prefix",
        )
        for key, value in source.model.state_dict().items():
            torch.testing.assert_close(target.model.state_dict()[key], value)

    def test_incomplete_full_weights_cannot_fall_back_to_inner_loading(self):
        model = tiny_model()
        model.register_buffer("metric_one", torch.tensor([1.0]))
        model.register_buffer("metric_two", torch.tensor([2.0]))
        state = model.state_dict()
        state.pop("metric_two")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.ckpt"
            torch.save(state, path)
            with self.assertRaisesRegex(RuntimeError, "full model"):
                load_ts_checkpoint(model, path, "cpu")

    def test_projection_only_and_incomplete_frozen_weights_are_rejected(self):
        for kind in (
            "projections_only",
            "missing_frozen",
            "wrong_shape",
            "missing_projection",
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                model = tiny_model()
                weights = model.state_dict()
                if kind == "projections_only":
                    weights = {
                        key: value for key, value in weights.items() if "project_modules" in key
                    }
                elif kind == "missing_frozen":
                    weights.pop("model.diffusion_model.base.bias")
                elif kind == "wrong_shape":
                    weights["model.diffusion_model.base.weight"] = torch.ones(3, 3)
                else:
                    weights.pop("model.diffusion_model.project_modules.bias")
                path = Path(folder) / "weights.ckpt"
                torch.save(weights, path)
                message = "every text projection" if kind == "missing_projection" else "full model"
                before = {key: value.clone() for key, value in model.state_dict().items()}
                with self.assertRaisesRegex(RuntimeError, message):
                    load_ts_checkpoint(model, path, "cpu")
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value)

    def test_unrelated_and_invalid_payloads_fail_before_assignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.ckpt"
            for payload, error in (
                ({"unrelated": torch.ones(2)}, RuntimeError),
                ([1, 2], TypeError),
            ):
                torch.save(payload, path)
                with self.assertRaises(error):
                    load_ts_checkpoint(tiny_model(), path, "cpu")

    def test_safetensors_and_lightning_share_state_reader(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as folder:
            state = {"weight": torch.arange(4).reshape(2, 2)}
            path = Path(folder) / "weights.safetensors"
            save_file(state, path)
            torch.testing.assert_close(load_state_dict(path)["weight"], state["weight"])


if __name__ == "__main__":
    unittest.main()
