import unittest

import torch

from nasmark.models.blocks import WatermarkModule
from nasmark.models.supernet import SPOSMobileNetSupernet, add_watermark_class


def tiny_model():
    return SPOSMobileNetSupernet(
        channels=(4, 4, 8, 8),
        downsample_layers=(1,),
        watermark_layers=(1, 2),
        watermark_channels=4,
    )


class ModelTests(unittest.TestCase):
    def test_canonical_spos_configuration(self):
        model = SPOSMobileNetSupernet()
        self.assertEqual(model.layers, 10)
        self.assertTrue(all(len(layer) == 4 for layer in model.candidate_blocks))
        self.assertEqual(tuple(model.watermark_modules.keys()), ("7", "8", "9"))

    def test_output_shape_and_shared_modules(self):
        model = tiny_model().eval()
        self.assertEqual(len(model.watermark_modules), 2)
        self.assertEqual(set(model.watermark_modules.keys()), {"1", "2"})
        self.assertTrue(all(isinstance(module, WatermarkModule) for module in model.watermark_modules.values()))
        # Four candidate blocks exist, but only one watermark module exists at layer 1.
        self.assertEqual(len(model.candidate_blocks[1]), 4)
        output = model(torch.randn(2, 3, 32, 32), (0, 1, 2))
        self.assertEqual(output.shape, (2, 10))

    def test_watermark_module_is_full_depthwise_bottleneck(self):
        module = tiny_model().watermark_modules["1"]
        convolutions = [item for item in module.modules() if isinstance(item, torch.nn.Conv2d)]
        self.assertEqual([item.kernel_size for item in convolutions], [(1, 1), (3, 3), (1, 1)])
        self.assertEqual(convolutions[1].groups, convolutions[1].in_channels)

    def test_exported_subnet_matches_supernet_path(self):
        model = tiny_model().eval()
        choice = (3, 1, 0)
        subnet = model.export_subnet(choice).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = model(images, choice)
            actual = subnet(images)
        torch.testing.assert_close(actual, expected)

    def test_watermark_class_preserves_original_logits(self):
        model = tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)
        choice = (0, 1, 2)
        with torch.no_grad():
            original = model(images, choice)
        add_watermark_class(model)
        with torch.no_grad():
            expanded = model(images, choice)
        self.assertEqual(expanded.shape, (2, 11))
        torch.testing.assert_close(expanded[:, :10], original)
        self.assertTrue(torch.all(expanded[:, 10] < 0))
        self.assertTrue(all(not p.requires_grad for p in model.classifier.base_classifier.parameters()))


if __name__ == "__main__":
    unittest.main()
