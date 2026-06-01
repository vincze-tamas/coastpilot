import unittest
from pipeline.recommendation import preference_score, recommendation_band


class TestRecommendation(unittest.TestCase):
    def test_rock_scores_higher_than_sand_same_conf(self):
        rock = preference_score('rock', 0.8, prefer_rock=1.0)
        sand = preference_score('sand', 0.8, prefer_rock=1.0)
        self.assertGreater(rock, sand)

    def test_band(self):
        self.assertEqual(recommendation_band(0.8), 'strong_match')
        self.assertEqual(recommendation_band(0.5), 'possible_match')
        self.assertEqual(recommendation_band(0.2), 'weak_match')


if __name__ == '__main__':
    unittest.main()
