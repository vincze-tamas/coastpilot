import unittest
from pipeline.ai.providers import _validate_structured_output


class TestAIProviders(unittest.TestCase):
    def test_validate_structured_output_invalid(self):
        out = _validate_structured_output({'label': 'BAD', 'confidence': None, 'rationale': ''})
        self.assertEqual(out['label'], 'unknown')

    def test_validate_structured_output_valid(self):
        out = _validate_structured_output({'label': 'sand', 'confidence': 0.91, 'rationale': 'ok'})
        self.assertEqual(out['label'], 'sand')
        self.assertGreaterEqual(out['confidence'], 0)
        self.assertLessEqual(out['confidence'], 1)


if __name__ == '__main__':
    unittest.main()
