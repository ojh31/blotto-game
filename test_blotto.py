import numpy as np
import unittest as ut
from blotto import PureStrat, BlottoGame


class TestBlotto3(ut.TestCase):

    def setUp(self):
        self.game = BlottoGame(num_soldiers=10, num_fields=3)
        self.binary_full_array = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.binary_position_array = np.array([2, 6])
        self.field_array = np.array([2, 3, 5])

    def testFromFields(self):
        strat = PureStrat(self.game, field_array=self.field_array)
        np.testing.assert_almost_equal(strat.binary_full_array,
                                       self.binary_full_array)
        self.assertTrue(True)

    def testFromPosition(self):
        binary_position_array = self.binary_position_array
        strat = PureStrat(self.game,
                          binary_position_array=binary_position_array)
        np.testing.assert_almost_equal(strat.binary_full_array,
                                       self.binary_full_array)
        self.assertTrue(True)

    def testFromFull(self):
        strat = PureStrat(self.game,
                          binary_full_array=self.binary_full_array)
        np.testing.assert_almost_equal(strat.field_array, self.field_array)
        self.assertTrue(True)


class TestBlotto4(ut.TestCase):

    def setUp(self):
        self.game = BlottoGame(num_soldiers=20, num_fields=4)
        self.strat_varied = PureStrat(self.game,
                                      field_array=np.array([0, 7, 7, 6]))
        self.strat_balanced = PureStrat(self.game,
                                        field_array=np.array([5, 5, 5, 5]))

    def test_battle(self):
        score1, score2 = self.strat_varied * self.strat_balanced
        self.assertEqual((score1, score2), (9, 1))

    def test_rank(self):
        self.assertTrue(self.strat_varied > self.strat_balanced)

if __name__ == "__main__":
    ut.main()
