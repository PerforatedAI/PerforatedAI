import unittest
from unittest import mock

import torch
import torch.nn as nn

from dendrite_variants.dendritron import dendritron
from perforatedai import modules_perforatedai as PA


class DendritronVariantTests(unittest.TestCase):
    def test_linear_parent_produces_matching_multidimensional_output(self):
        parent = nn.Linear(7, 5, bias=False).to(dtype=torch.float64)
        candidate = dendritron.create_dendritron_dendrite(parent)
        inputs = torch.randn(3, 4, 7, dtype=torch.float64, requires_grad=True)

        output = candidate(inputs)
        output.square().mean().backward()

        self.assertEqual(output.shape, (3, 4, 5))
        self.assertEqual(candidate.router.weight.dtype, parent.weight.dtype)
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(candidate.router.weight.grad)

    def test_routing_keeps_exactly_top_k_specialists(self):
        candidate = dendritron.DendritronLinear(8, 8, branches=4, top_k=2)
        candidate(torch.randn(6, 8))

        nonzero = (candidate.last_sparse_weights > 0).sum(dim=-1)
        self.assertTrue(torch.equal(nonzero, torch.full_like(nonzero, 2)))
        self.assertTrue(
            torch.allclose(candidate.last_sparse_weights.sum(dim=-1), torch.ones(6))
        )

    def test_factory_recreates_platform_best_candidate_configuration(self):
        candidate = dendritron.DendritronLinear(
            9, 6, branches=5, top_k=3, hidden_features=11, bias=False
        )

        recreated = dendritron.create_dendritron_dendrite(candidate)

        self.assertEqual(recreated.in_features, 9)
        self.assertEqual(recreated.out_features, 6)
        self.assertEqual(recreated.branches, 5)
        self.assertEqual(recreated.top_k, 3)
        self.assertEqual(recreated.hidden_features, 11)
        self.assertFalse(recreated.use_bias)

    def test_initializer_registers_a_configured_factory(self):
        tracker = mock.Mock()
        with mock.patch.object(dendritron.GPA, "pai_tracker", tracker):
            dendritron.initialize_variant_dendrite(
                branches=3, top_k=1, hidden_features=12
            )

        create_fn = tracker.set_create_dendrite_global.call_args.args[0]
        candidate = create_fn(nn.Linear(4, 2))
        self.assertEqual(candidate.branches, 3)
        self.assertEqual(candidate.top_k, 1)
        self.assertEqual(candidate.hidden_features, 12)

    def test_factory_rejects_non_linear_modules(self):
        with self.assertRaises(TypeError):
            dendritron.create_dendritron_dendrite(nn.Conv2d(1, 2, 3))

    def test_constructor_rejects_invalid_routing_configuration(self):
        with self.assertRaises(ValueError):
            dendritron.DendritronLinear(4, 4, branches=2, top_k=3)
        with self.assertRaises(ValueError):
            dendritron.DendritronLinear(4, 4, hidden_features=0)

    def test_checkpoint_reconstruction_preserves_variant_factory(self):
        tracker = mock.Mock()
        tracker.member_vars = {"optimizer_instance": None}
        with mock.patch.object(PA.GPA, "pai_tracker", tracker):
            wrapped = PA.PAINeuronModule(nn.Linear(4, 4), ".test")
        wrapped.set_create_dendrite(dendritron.create_dendritron_dendrite)

        wrapped.clear_dendrites()
        candidate = wrapped.dendrite_module.create_dendrite(wrapped.main_module)

        self.assertIsInstance(candidate, dendritron.DendritronLinear)


if __name__ == "__main__":
    unittest.main()
