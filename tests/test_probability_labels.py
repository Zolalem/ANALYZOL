import unittest
import numpy as np
from analyz_ol_v8 import PoissonDC


class ProbabilityLabelTests(unittest.TestCase):
    def test_home_draw_away_labels_match_score_matrix(self):
        model = PoissonDC.__new__(PoissonDC)
        matrix = np.array([
            [0.10, 0.20],  # 0 home goals
            [0.30, 0.40],  # 1 home goal
        ])
        model.score_matrix = lambda home, away, max_goals=10: matrix
        model.expected_goals = lambda home, away: (1.0, 1.0)

        result = model.predict("Home", "Away")

        self.assertAlmostEqual(result["p_home"], 0.30)
        self.assertAlmostEqual(result["p_draw"], 0.50)
        self.assertAlmostEqual(result["p_away"], 0.20)
        self.assertAlmostEqual(
            result["p_home"] + result["p_draw"] + result["p_away"], 1.0
        )

    def test_home_and_away_are_not_reversed(self):
        model = PoissonDC.__new__(PoissonDC)
        matrix = np.array([
            [0.05, 0.15, 0.10],
            [0.20, 0.10, 0.05],
            [0.20, 0.10, 0.05],
        ])
        model.score_matrix = lambda home, away, max_goals=10: matrix
        model.expected_goals = lambda home, away: (1.0, 1.0)

        result = model.predict("Home", "Away")

        self.assertAlmostEqual(result["p_home"], 0.30)
        self.assertAlmostEqual(result["p_draw"], 0.20)
        self.assertAlmostEqual(result["p_away"], 0.50)


if __name__ == "__main__":
    unittest.main()
