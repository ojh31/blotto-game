import numpy as np
import itertools
from collections import defaultdict

# questions:
# average score or total wins?
# draws split points?


class BlottoLearn(object):
    # Plays repeated tournaments and learns via genetic algorithm

    def __init__(self, blottoGame, num_players, num_iterations):
        self.blottoGame = blottoGame
        self.num_players = num_players
        self.num_iterations = num_iterations
        self.strat_list = [blottoGame.rand_strat()
                           for i in range(self.num_players)]

    def iterate(self):
        # perform iterative improvement step
        tour = BlottoTour(self.strat_list)
        tour_size = self.num_players
        strats = tour.sort()
        strats = strats[:int(0.2 * tour_size)]
        offspring = [strat.add_noise()
                     for _ in range(3)
                     for strat in strats[:int(0.05 * tour_size)]]
        strats += offspring
        to_add = self.num_players - len(strats)
        strats += [self.blottoGame.rand_strat() for _ in range(to_add)]
        self.strat_list = strats

    def learn(self):
        # perform all iterations and publish leaderboard

class BlottoTour(object):
    # A Blotto round-robin tournament

    def __init__(self, strat_list):
        self.strat_list = strat_list

    def run_tour(self):
        strat_list = self.strat_list
        pairs = itertools.combinations(strat_list, 2)
        score_dict = defaultdict(lambda: 0)
        for strat1, strat2 in pairs:
            score1, score2 = strat1 * strat2
            score_dict[strat1] += score1
            score_dict[strat2] += score2
        return score_dict

    def sort(self):
        # sort by performance in tournament
        score_dict = self.run_tour()
        scores = sorted(score_dict, key=score_dict.get)
        return scores


class BlottoGame(object):
    # A heterogeneous Blotto game in which the nth field has n points available

    def __init__(self, num_soldiers, num_fields):
        self.num_soldiers = num_soldiers
        self.num_fields = num_fields
        self.num_zeros = self.num_soldiers
        self.num_ones = self.num_fields - 1
        self.num_digits = self.num_zeros + self.num_ones

    def total_points_available(self):
        # sum of points over all the fields
        n = self.num_fields
        return (n * (n + 1)) / 2

    def position_to_field_array(self, binary_position_array):
        # convert binary_position_array to field_array
        first_field = binary_position_array[0] + 1
        last_field = self.num_digits - binary_position_array[-1]
        field_array = np.ediff1d(binary_position_array,
                                 to_begin=first_field,
                                 to_end=last_field) - 1
        return field_array

    def position_to_full_array(self, binary_position_array):
        # convert binary_position_array to binary_full_array
        np.testing.assert_almost_equal(binary_position_array,
                                       np.sort(binary_position_array))
        assert binary_position_array.size == self.num_ones
        binary_full_array = np.zeros(self.num_digits)
        binary_full_array[binary_position_array] = 1
        return binary_full_array

    def field_to_position_array(self, field_array):
        # convert field_array to binary_position_array
        assert field_array.size == self.num_fields
        binary_position_array = (np.cumsum(field_array) +
                                 np.arange(self.num_fields)
                                 )
        binary_position_array = binary_position_array[:-1]
        return binary_position_array

    def full_to_position_array(self, binary_full_array):
        # convert binary_full_array to binary_position_array
        zeros = binary_full_array == 0
        ones = binary_full_array == 1
        binary = zeros | ones
        assert binary.all()
        assert binary_full_array.size == self.num_digits
        (binary_position_array,) = np.where(binary_full_array == 1)
        return binary_position_array

    def rand_strat(self):
        # Generates a random pure strategy
        # as a vector of num_soldiers 0s and
        #                num_fields - 1 1s
        num_zeros = self.num_zeros
        num_ones = self.num_ones
        binary_array_length = num_zeros + num_ones
        binary_position_array = np.random.choice(binary_array_length,
                                                 num_ones,
                                                 replace=False)
        binary_position_array = np.sort(binary_position_array)
        binary_full_array = self.position_to_full_array(binary_position_array)
        return PureStrat(self, binary_full_array=binary_full_array)


class PureStrat(object):
    # A pure strategy for a game of heterogeneous Blotto

    def __init__(self, blottoGame, **kwargs):
        # Sum of soldiers over flags must equal num_soldiers
        # Can specify by array of soliders per field
        # or as a binary string
        self.blottoGame = blottoGame
        if len(kwargs) != 1:
            raise IOError(kwargs)
        array_type = kwargs.keys()[0]
        array = kwargs.values()[0]
        if array_type == "binary_position_array":
            binary_position_array = array
            field_array = \
                blottoGame.position_to_field_array(binary_position_array)
            binary_full_array = \
                blottoGame.position_to_full_array(binary_position_array)
        elif array_type == "binary_full_array":
            binary_full_array = array
            binary_position_array = \
                blottoGame.full_to_position_array(binary_full_array)
            field_array = \
                blottoGame.position_to_field_array(binary_position_array)
        elif array_type == "field_array":
            field_array = array
            binary_position_array = \
                blottoGame.field_to_position_array(field_array)
            binary_full_array = \
                blottoGame.position_to_full_array(binary_position_array)
        else:
            raise IOError(array_type)
        self.binary_full_array = binary_full_array
        self.binary_position_array = binary_position_array
        self.field_array = field_array

    def add_noise(self):
        # create similar strategy based on binary string
        binary_full_array = self.binary_full_array.copy()
        (ones,) = np.where(binary_full_array == 1)
        rem_loc = np.random.choice(ones)
        binary_full_array = np.delete(binary_full_array, rem_loc)
        add_loc = np.random.choice(self.blottoGame.num_digits)
        binary_full_array = np.insert(binary_full_array, add_loc, 1)
        return PureStrat(self.blottoGame, binary_full_array=binary_full_array)

    def get_offspring(self, num_kids):
        # gets a random list of related strategies
        return [self.add_noise() for i in range(num_kids)]

    def __mult__(self, other):
        # overload * operator to face off between 2 strats in the same game
        assert self.blottoGame == other.blottoGame
        flag_points = np.arange(self.blottoGame.num_flags)
        wins = self.field_array > other.field_array
        draws = self.field_array == other.field_array
        losses = self.field_array < other.field_array
        my_points = wins * flag_points + 0.5 * draws * flag_points
        your_points = 0.5 * draws * flag_points + losses * flag_points
        total_points = my_points + your_points
        assert total_points == self.blottoGame.total_points_available()
        return my_points, your_points

    def __lt__(self, other):
        # tests if self loses to other in face off
        my_points, your_points = self * other
        return my_points < your_points

    def __gt__(self, other):
        # tests if self beats other in face off
        my_points, your_points = self * other
        return my_points > your_points

    def __le__(self, other):
        # tests if self loses or draws to other in face off
        my_points, your_points = self * other
        return my_points <= your_points

    def __ge__(self, other):
        # tests if self beats or draws to other in face off
        my_points, your_points = self * other
        return my_points >= your_points
