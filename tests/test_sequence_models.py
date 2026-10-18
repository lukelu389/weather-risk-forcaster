"""Actual training, frozen preprocessing, recurrence, and checkpoint checks."""

import copy
import io
import json
import unittest

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "torch"}:
        raise
    raise unittest.SkipTest("Phase 4 tests require the optional sequence dependencies") from exc

from weather_risk.sequence_models import (
    _DiagonalStateSpace, _GRUVariance, fit_sequence_variance, predict_sequence_variance,
)


class SequenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        signal = np.linspace(-1.2, 1.2, 100)
        cls.X = np.zeros((100, 5, 2))
        cls.X[:, :, 0] = signal[:, None]
        cls.X[:, :, 1] = np.random.default_rng(42).normal(size=(100, 5))
        cls.y = np.exp(signal) * np.where(np.arange(100) % 2, 1, -1)
        cls.fitted = {
            architecture: fit_sequence_variance(cls.X, cls.y, architecture=architecture,
                                                hidden_size=hidden, epochs=45)
            for architecture, hidden in (("gru", 8), ("ssm", 32))
        }

    def test_both_models_learn_heteroskedastic_signal_with_fixed_budget(self):
        for architecture, (metadata, checkpoint) in self.fitted.items():
            with self.subTest(architecture=architecture):
                variance = predict_sequence_variance(self.X, metadata, checkpoint)
                self.assertEqual(variance.shape, (len(self.y),))
                self.assertTrue(np.isfinite(variance).all())
                self.assertTrue((variance > 0).all())
                self.assertLess(metadata["loss_final"], metadata["loss_initial"] - .2)
                self.assertGreater(variance[-20:].mean(), variance[:20].mean() * 5)
                normalized = variance / metadata["target_scale"] ** 2
                measured_loss = .5 * np.mean(np.log(normalized)
                                             + (self.y / metadata["target_scale"]) ** 2 / normalized)
                self.assertAlmostEqual(metadata["loss_final"], measured_loss, places=13)
                self.assertEqual(metadata["epochs"], 45)
                self.assertEqual(metadata["optimizer_steps"], 45)
                self.assertEqual(metadata["checkpoint_epoch"], 45)
                self.assertEqual(metadata["checkpoint_selection"], "final_epoch")
                self.assertEqual(len(metadata["loss_history"]), 46)
                self.assertEqual(metadata["loss_initial"], metadata["loss_history"][0])
                self.assertEqual(metadata["loss_final"], metadata["loss_history"][-1])
                self.assertEqual(metadata["parameter_count"], sum(value.numel() for value in checkpoint.values()))
                self.assertEqual(metadata["batch_size"], len(self.y))
                self.assertTrue(all(value.device.type == "cpu" and value.dtype == torch.float64
                                    for value in checkpoint.values()))
                json.dumps(metadata, allow_nan=False)

    def test_fixed_seed_repeats_exactly_and_different_seed_changes_training(self):
        for architecture in ("gru", "ssm"):
            with self.subTest(architecture=architecture):
                args = dict(architecture=architecture, hidden_size=4, epochs=4)
                first, first_checkpoint = fit_sequence_variance(self.X[:20], self.y[:20], **args)
                repeat, repeat_checkpoint = fit_sequence_variance(self.X[:20], self.y[:20], **args)
                changed, changed_checkpoint = fit_sequence_variance(self.X[:20], self.y[:20], seed=1730, **args)
                self.assertEqual(first, repeat)
                self.assertTrue(all(torch.equal(first_checkpoint[name], repeat_checkpoint[name])
                                    for name in first_checkpoint))
                self.assertNotEqual(first["loss_history"], changed["loss_history"])
                self.assertTrue(any(not torch.equal(first_checkpoint[name], changed_checkpoint[name])
                                    for name in first_checkpoint))

    def test_metadata_and_tensor_serialization_restore_identical_predictions(self):
        for architecture, (metadata, checkpoint) in self.fitted.items():
            with self.subTest(architecture=architecture):
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=True, map_location="cpu")
                saved_metadata = json.loads(json.dumps(metadata, allow_nan=False))
                np.testing.assert_array_equal(predict_sequence_variance(self.X, metadata, checkpoint),
                                              predict_sequence_variance(self.X, saved_metadata, restored))

    def test_missing_features_train_only_preprocessing_and_batch_independence(self):
        X = np.empty((4, 3, 2))
        X[:, :, 0] = np.arange(12).reshape(4, 3)
        X[:, :, 1] = np.nan
        X[0, 0, 0] = np.nan
        for architecture in ("gru", "ssm"):
            with self.subTest(architecture=architecture):
                metadata, checkpoint = fit_sequence_variance(X, [1., -2., 3., -4.],
                                                             architecture=architecture, hidden_size=4, epochs=3)
                frozen = copy.deepcopy(metadata)
                self.assertEqual(metadata["medians"], [6., 0.])
                self.assertEqual(metadata["preprocessing_rows"], 12)
                self.assertEqual(metadata["centers"][1:], [0., 1. / 12, 1.])
                self.assertEqual(metadata["scales"][1], 1.)
                self.assertEqual(metadata["scales"][3], 1.)
                current = np.array([[[np.nan, 7.], [15., np.nan], [16., 8.]]])
                future = np.full((2, 3, 2), 1e6)
                single = predict_sequence_variance(current, metadata, checkpoint)
                batch = predict_sequence_variance(np.concatenate((current, future)), metadata, checkpoint)
                np.testing.assert_allclose(single, batch[:1], rtol=1e-13, atol=1e-13)
                self.assertTrue((batch > 0).all())
                self.assertEqual(metadata, frozen)

    def test_target_scaling_returns_variance_in_original_units(self):
        for architecture in ("gru", "ssm"):
            with self.subTest(architecture=architecture):
                args = dict(architecture=architecture, hidden_size=4, epochs=5)
                original, checkpoint = fit_sequence_variance(self.X[:20], self.y[:20], **args)
                scaled, scaled_checkpoint = fit_sequence_variance(self.X[:20], self.y[:20] * 10, **args)
                np.testing.assert_allclose(predict_sequence_variance(self.X, scaled, scaled_checkpoint),
                                           predict_sequence_variance(self.X, original, checkpoint) * 100,
                                           rtol=1e-12)

    def test_models_are_sensitive_to_sequence_order(self):
        # Equal gates in the GRU and a 0.5 SSM decay expose the actual recurrence;
        # a flattened mean or last-independent placeholder would fail this check.
        values = torch.tensor([[[1.], [0.], [0.]], [[0.], [0.], [1.]]], dtype=torch.float64)
        for model in (_GRUVariance(1, 1), _DiagonalStateSpace(1, 1)):
            with self.subTest(model=type(model).__name__), torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.head.weight.fill_(1.)
                if isinstance(model, _GRUVariance):
                    model.recurrent.weight_ih_l0[2, 0] = 1.
                else:
                    model.input_projection.weight.fill_(1.)
                predicted = model(values).numpy()
                self.assertGreater(predicted[1], predicted[0])
                self.assertGreater(predicted[0], 0.)

    def test_state_space_recurrence_matches_hand_calculation_and_is_stable(self):
        model = _DiagonalStateSpace(1, 1)
        with torch.no_grad():
            model.log_decay.zero_()
            model.input_projection.weight.fill_(1.)
            model.input_projection.bias.zero_()
            model.head.weight.fill_(1.)
            model.head.bias.zero_()
            # h = 1, 2.5, 4.25 for inputs 1, 2, 3 and diagonal transition 0.5.
            actual = model(torch.tensor([[[1.], [2.], [3.]]], dtype=torch.float64)).item()
            self.assertAlmostEqual(actual, 12 * np.tanh(4.25 / 12), places=14)
            impulse = torch.zeros((1, 21, 1), dtype=torch.float64)
            impulse[0, 0, 0] = 1.
            final = model(impulse).item()
            self.assertAlmostEqual(final, 12 * np.tanh(0.5**20 / 12), places=15)
            model.log_decay.fill_(1000.)
            self.assertLess(model.decay().item(), 1.)
            self.assertGreaterEqual(model.decay().item(), 0.)

    def test_smooth_variance_bound_keeps_large_finite_inputs_positive(self):
        extreme = np.full((2, self.X.shape[1], self.X.shape[2]), 1e6)
        for architecture, (metadata, checkpoint) in self.fitted.items():
            with self.subTest(architecture=architecture):
                result = predict_sequence_variance(extreme, metadata, checkpoint)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue((result > 0).all())
                self.assertTrue((result <= np.exp(12) * metadata["target_scale"] ** 2).all())
                self.assertTrue((result >= np.exp(-12) * metadata["target_scale"] ** 2).all())

    def test_prediction_rejects_corrupt_or_incompatible_checkpoints(self):
        metadata, checkpoint = self.fitted["gru"]
        invalid_checkpoints = []
        missing = dict(checkpoint)
        missing.pop(next(iter(missing)))
        invalid_checkpoints.append(missing)
        for replacement in (torch.zeros(2, dtype=torch.float64),
                            checkpoint["head.bias"].float(),
                            torch.full_like(checkpoint["head.bias"], float("nan"))):
            changed = dict(checkpoint)
            changed["head.bias"] = replacement
            invalid_checkpoints.append(changed)
        for changed in invalid_checkpoints:
            with self.subTest(checkpoint=list(changed)), self.assertRaisesRegex(ValueError, "[Cc]heckpoint"):
                predict_sequence_variance(self.X, metadata, changed)

    def test_prediction_rejects_wrong_shapes_and_invalid_metadata(self):
        metadata, checkpoint = self.fitted["gru"]
        for X in (np.ones((2, 4, 2)), np.ones((2, 5, 3)), np.ones((2, 5)), np.full((2, 5, 2), np.inf)):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                predict_sequence_variance(X, metadata, checkpoint)
        for field, value in (("schema_version", 99), ("scales", [0.] * 4), ("target_scale", -1.),
                             ("medians", [1.]), ("mean", "estimated"), ("architecture", "mamba")):
            changed = dict(metadata)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                predict_sequence_variance(self.X, changed, checkpoint)
        self.assertEqual(predict_sequence_variance(self.X[:0], metadata, checkpoint).shape, (0,))

    def test_invalid_training_inputs_and_budgets_fail(self):
        X = np.ones((4, 3, 2))
        y = np.arange(1., 5.)
        args = dict(architecture="gru", hidden_size=2, epochs=1)
        invalid_data = ((X[0], y), (X[:, :0], y), (X[:, :, :0], y),
                        (np.full_like(X, np.inf), y), (X, y[:-1]), (X, np.zeros(4)),
                        (X, [1., 2., np.nan, 4.]), (X, np.ones((4, 1))), (X[:1], y[:1]),
                        (X, np.full(4, 1e308)))
        for values, targets in invalid_data:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                fit_sequence_variance(values, targets, **args)
        for field, value in (("architecture", "mamba"), ("hidden_size", 0), ("hidden_size", 2.5),
                             ("epochs", 0), ("epochs", True), ("learning_rate", 0.),
                             ("learning_rate", float("nan")), ("seed", -1), ("seed", 2**63)):
            changed = dict(args)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fit_sequence_variance(X, y, **changed)


if __name__ == "__main__":
    unittest.main()
