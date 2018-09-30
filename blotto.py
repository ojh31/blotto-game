import numpy as np
import itertools
from collections import defaultdict
import operator
import argparse

# questions:
# average score or total wins?
# draws split points?
mutate_range = 0.05
keep_range = 0.2


class BlottoLearn(object):
    # Plays repeated tournaments and learns via genetic algorithm

    def __init__(self, blottoGame, num_players):
        self.blottoGame = blottoGame
        self.num_players = num_players
        self.strat_list = [blottoGame.uniform_strat()
                           for i in range(self.num_players)]

    def tour(self):
        # BlottoTournament object for current strategies
        return BlottoTour(self.blottoGame, self.num_players, self.strat_list)

    def iterate(self):
        # perform iterative improvement step
        tour = self.tour()
        tour_size = self.num_players
        strats = tour.sort()
        strats = strats[:int(keep_range * tour_size)]
        offspring = [strat.add_noise()
                     for _ in range(3)
                     for strat in strats[:int(mutate_range * tour_size)]]
        strats += offspring
        to_add = self.num_players - len(strats)
        strats += [self.blottoGame.uniform_strat() for _ in range(to_add)]
        self.strat_list = strats

    def learn(self, num_iterations):
        # perform all iterations and publish leaderboard
        (self.iterate() for _ in range(num_iterations))

    def score_board(self):
        # score board for current strategies list
        return self.tour().score_board()


class BlottoTour(object):
    # A Blotto round-robin tournament

    def __init__(self, blottoGame, num_players, strat_list):
        assert num_players == len(strat_list)
        self.blottoGame = blottoGame
        self.num_players = num_players
        self.strat_list = strat_list

    def max_score(self):
        # Maximum possible points per player
        return (self.blottoGame.total_points_available() *
                (self.num_players - 1))

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
        scores = sorted(score_dict, key=score_dict.get, reverse=True)
        return scores

    def score_board(self):
        # run tournament and show scores in order
        score_dict = self.run_tour()
        score_board = sorted(score_dict.items(),
                             key=operator.itemgetter(1),
                             reverse=True)
        score_board = [(strat, score / self.max_score())
                       for strat, score in score_board]
        return score_board


class BlottoGame(object):
    # A heterogeneous Blotto game in which the nth field has n points available

    def __init__(self, num_soldiers, num_fields, distribution='multinomial'):
        self.num_soldiers = num_soldiers
        self.num_fields = num_fields
        self.num_zeros = self.num_soldiers
        self.num_ones = self.num_fields - 1
        self.num_digits = self.num_zeros + self.num_ones
        self.distribution = distribution

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

    def uniform_strat(self):
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

    def multinomial_strat(self):
        # generates multinomial random strategy
        points_available = self.total_points_available
        probs = [(i + 1) / points_available for i in range(self.num_fields)]
        field_array = np.random.multinomial(self.num_soldiers, probs)
        return PureStrat(self, field_array=field_array)

    def rand_strat(self):
        # generate random strategy
        if self.distribution == 'uniform':
            return self.uniform_strat()
        elif self.distribution == 'multinomial':
            return self.multinomial_strat()
        else:
            raise AttributeError('bad distribution')


class PureStrat(object):
    # A pure strategy for a game of heterogeneous Blotto

    def __init__(self, blottoGame, **kwargs):
        # Sum of soldiers over fields must equal num_soldiers
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

    def __mul__(self, other):
        # overload * operator to face off between 2 strats in the same game
        assert self.blottoGame == other.blottoGame
        flag_points = np.arange(self.blottoGame.num_fields) + 1
        wins = self.field_array > other.field_array
        draws = self.field_array == other.field_array
        losses = self.field_array < other.field_array
        my_points = (np.dot(wins, flag_points) +
                     0.5 * np.dot(draws, flag_points))
        your_points = (np.dot(losses, flag_points) +
                       0.5 * np.dot(draws, flag_points))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customise genetic algorithm')
    parser.add_argument('num_iterations', metavar='n_iterations', type=int,
                        help=('The number of iterations to use in'
                              'learning algorithm'))
    parser.add_argument('num_players', metavar='n_players', type=int,
                        help=('The number of players to use in'
                              'round robin tournaments'))
    parser.add_argument('--num_soldiers', type=int, default=100,
                        help=('The number of soldiers to use in'
                              'Blotto Game'))
    parser.add_argument('--num_fields', type=int, default=10,
                        help=('The number of battlefields to use in'
                              'Blotto Game'))
    parser.add_argument('--uniform', dest='distribution',
                        default='multinomial',
                        help='Generate random strategies uniformly')
    args = parser.parse_args()
    bg = BlottoGame(num_soldiers=args.num_soldiers,
                    num_fields=args.num_fields,
                    distribution=args.distribution)
    bl = BlottoLearn(bg, num_players=args.num_players)
    bl.learn(num_iterations=args.num_iterations)
    with open('scores.txt', 'w') as f:
        scores = bl.score_board()
        output_rows = ['Strat: %s Score: %.3f' %
                       (str(strat.field_array), score)
                       for strat, score in scores]
        output = '\n'.join(output_rows)
        f.write(output)
