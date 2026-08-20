import copy
import unittest

import torch

from nasmark.data import build_synthetic_loaders
from nasmark.search import random_search
from nasmark.models.supernet import add_watermark_class
from nasmark.training import (
    configure_stage2_parameters,
    estimate_contribution_scores,
    select_low_contribution,
    train_stage1_epoch,
    train_stage2_epoch,
)
from tests.test_model import tiny_model


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = tiny_model()
        self.loaders = build_synthetic_loaders(batch_size=2, samples=4)

    def test_stage1_updates_watermark_from_the_beginning(self):
        before = [parameter.detach().clone() for parameter in self.model.watermark_modules.parameters()]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        train_stage1_epoch(self.model, self.loaders["train"], optimizer, "cpu")
        after = list(self.model.watermark_modules.parameters())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after)))

    def test_stage2_parameter_selection_and_training(self):
        add_watermark_class(self.model)
        scores = estimate_contribution_scores(self.model, self.loaders["train"], "cpu", 1)
        selected = select_low_contribution(scores, 0.25)
        trainable = configure_stage2_parameters(self.model, selected)
        names = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.assertTrue(set(selected).issubset(names))
        self.assertTrue(any(name.startswith("watermark_modules.") for name in names))
        self.assertTrue(any(name.startswith("classifier.watermark_classifier.") for name in names))
        self.assertFalse(any(name.startswith("classifier.base_classifier.") for name in names))
        optimizer = torch.optim.SGD(trainable, lr=0.01)
        forward_calls = []
        hook = self.model.register_forward_hook(lambda *_: forward_calls.append(1))
        metrics = train_stage2_epoch(
            self.model,
            self.loaders["train"],
            self.loaders["trigger_train"],
            optimizer,
            "cpu",
        )
        hook.remove()
        self.assertIn("watermark_accuracy", metrics)
        self.assertEqual(len(forward_calls), len(self.loaders["train"]))

    def test_search_does_not_change_weights(self):
        add_watermark_class(self.model)
        before = copy.deepcopy(self.model.state_dict())
        results = random_search(
            self.model,
            self.loaders["valid"],
            self.loaders["trigger_valid"],
            "cpu",
            samples=2,
        )
        self.assertEqual(len(results), 2)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name])


if __name__ == "__main__":
    unittest.main()
