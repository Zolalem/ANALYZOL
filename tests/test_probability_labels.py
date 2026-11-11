import numpy as np
from analyz_ol_v8 import PoissonDC


def test_home_win_mass_comes_from_lower_triangle():
    model = PoissonDC()
    matrix = np.zeros((3, 3), dtype=float)
    matrix[2, 0] = 0.60  # home scores more: home win
    matrix[1, 1] = 0.25  # draw
    matrix[0, 2] = 0.15  # away scores more: away win
    model.score_matrix = lambda home, away, max_goals=10: matrix
    model.expected_goals = lambda home, away: (1.5, 1.0)

    prediction = model.predict("Home", "Away")

    assert prediction["p_home"] == 0.60
    assert prediction["p_draw"] == 0.25
    assert prediction["p_away"] == 0.15
    assert abs(
        prediction["p_home"] + prediction["p_draw"] + prediction["p_away"] - 1.0
    ) < 1e-12
