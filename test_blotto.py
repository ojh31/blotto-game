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

if __name__ == "__main__":
    ut.main()
