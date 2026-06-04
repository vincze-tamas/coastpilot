import unittest

from pipeline.config import PipelineConfig


class TestPipelineConfig(unittest.TestCase):
    def test_review_calibrated_default_rock_threshold(self):
        self.assertEqual(PipelineConfig().rock_min_confidence, 0.80)


if __name__ == '__main__':
    unittest.main()
