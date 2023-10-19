# Simple, generic Metropolis-Hastings MCMC sampler

import numpy as np
np.set_printoptions(precision=2)
from jkit.math_util import sample_categorical
from jkit import progressbar
from collections import namedtuple

TerminalColors = namedtuple("TerminalColors", ["blue", "green", "yellow", "red", "default"])
terminal_colors = TerminalColors(
    blue = '\033[34m',
    green = '\033[32m',
    yellow = '\033[33m',
    red = '\033[31m',
    default = '\033[0m',
)
class Move(object):
    '''
    Generic MCMC Move class.
    '''

    def __init__(self):
        return NotImplemented

    def p(self, state, data):
        '''
        Probability (or function proportional to it) that this
        move type will be chosen given the current state.
        '''

        return NotImplemented

    def propose(self, state):
        '''
        Proposal of a change to the current state.
        '''

        return NotImplemented

    def accept(self, state, change):
        '''
        Apply the given change to the state to generate a new state
        '''
        return NotImplemented

    def log_alpha(self, state, change):
        '''
        Logarithm of the acceptance ratio for the Metropolis
        acceptance criterion
        '''
        return NotImplemented

class MCMCSampler(object):
    def __init__(self, moves):
        self.num_stages = len(moves)
        self.moves = dict()
        for i,stage in enumerate(moves):
            self.moves[i] = dict()
            for mv in stage:
                self.moves[i][mv.name] = mv
        self.progressbar = progressbar.IncrementalBar(prefix="")

    def sample(self, n, state_0, data, verbose=False):
        accepts = np.zeros(self.num_stages)
        state = state_0.copy()
        samples = [state]
        for i in self.progressbar.iter(range(1,n)):
            if verbose:
                print i
            for stage in self.moves.keys():

                # Calculate the probability of performing each move
                move_p = []
                move_list = []
                for mv_name, mv in self.moves[stage].iteritems():
                    mv_p, mv_par = mv.p(state, data)
                    move_p += mv_p
                    move_list += mv_par
                move_p = np.array(move_p)

                # Sample the move type to make
                typ = sample_categorical(move_p)

                # Sample the change from the proposal distribution
                change = move_list[typ]

                # Calculate the acceptance ratio of that change
                log_alpha = self.moves[stage][change[0]].log_alpha(state, change, data)

                if verbose:
                    s = "Chose \n{}\nwith probability {}".format(change, move_p[typ])
                    print terminal_colors.blue+s+terminal_colors.default

                # Apply Metropolis criterion
                if np.log(np.random.random()) < log_alpha:
                    # Accept the change, and transition to the new state
                    if verbose:
                        print terminal_colors.green+"Accept!"+terminal_colors.default
                    state = self.moves[stage][change[0]].accept(state, change)
                    accepts[stage] += 1
                else:
                    # reject the change and keep the current state
                    if verbose:
                        print terminal_colors.red+"Reject!"+terminal_colors.default

                if verbose:
                    print state

            samples.append(state.copy())

        return samples, accepts/n

class State(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def copy(self):
        return self.__class__(**(self.__dict__))

    def __repr__(self):
        out_str = "-"*32+"( State )"+"-"*32+"\n"
        for k, v in self.__dict__.iteritems():
            out_str += k + " = " + str(v) + "\n"
        out_str += "-"*73+"\n"
        return out_str

    def __str__(self):
        return self.__repr__()
