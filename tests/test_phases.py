import unittest
from dataclasses import dataclass
from assyst.phases import refine_concentration_jumps, ConcentrationJump

@dataclass
class MockPoint:
    c: float
    mu: float
    T: float

class MockPhase:
    def __init__(self, jump_mu=0.5, c_low=0.1, c_high=0.9):
        self.jump_mu = jump_mu
        self.c_low = c_low
        self.c_high = c_high

    def concentration(self, mu, T):
        if mu < self.jump_mu:
            return self.c_low
        else:
            return self.c_high

class TestPhases(unittest.TestCase):
    def test_single_jump(self):
        phase = MockPhase(jump_mu=0.5, c_low=0.2, c_high=0.8)
        points = [
            MockPoint(c=0.2, mu=0.0, T=300),
            MockPoint(c=0.2, mu=0.4, T=300),
            MockPoint(c=0.8, mu=0.6, T=300),
            MockPoint(c=0.8, mu=1.0, T=300),
        ]

        jumps = refine_concentration_jumps(points, phase, jump_threshold=0.5, mu_tolerance=1e-5)

        self.assertEqual(len(jumps), 1)
        jump = jumps[0]
        self.assertAlmostEqual(jump.mu, 0.5, delta=1e-4)
        self.assertAlmostEqual(jump.c_left, 0.2)
        self.assertAlmostEqual(jump.c_right, 0.8)
        self.assertEqual(jump.T, 300)

    def test_no_jump(self):
        phase = MockPhase(jump_mu=0.5, c_low=0.2, c_high=0.8)
        points = [
            MockPoint(c=0.2, mu=0.0, T=300),
            MockPoint(c=0.2, mu=0.2, T=300),
            MockPoint(c=0.2, mu=0.4, T=300),
        ]

        jumps = refine_concentration_jumps(points, phase, jump_threshold=0.5)
        self.assertEqual(len(jumps), 0)

    def test_multiple_temperatures(self):
        phase = MockPhase(jump_mu=0.5, c_low=0.1, c_high=0.9)
        points = [
            # T=300
            MockPoint(c=0.1, mu=0.4, T=300),
            MockPoint(c=0.9, mu=0.6, T=300),
            # T=400
            MockPoint(c=0.1, mu=0.4, T=400),
            MockPoint(c=0.9, mu=0.6, T=400),
        ]

        jumps = refine_concentration_jumps(points, phase, jump_threshold=0.5)
        self.assertEqual(len(jumps), 2)
        self.assertEqual(jumps[0].T, 300)
        self.assertEqual(jumps[1].T, 400)
        for jump in jumps:
            self.assertAlmostEqual(jump.mu, 0.5, delta=1e-4)

    def test_dict_points(self):
        phase = MockPhase(jump_mu=0.5, c_low=0.2, c_high=0.8)
        points = [
            {'c': 0.2, 'mu': 0.0, 'T': 300},
            {'c': 0.8, 'mu': 1.0, 'T': 300},
        ]
        jumps = refine_concentration_jumps(points, phase, jump_threshold=0.5)
        self.assertEqual(len(jumps), 1)
        self.assertAlmostEqual(jumps[0].mu, 0.5, delta=1e-4)

if __name__ == '__main__':
    unittest.main()
