import os
os.chdir('../')

from distribution import Distribution, Constant, Gaussian, Choose, SampledDistribution
from itertools import product
from collections import defaultdict
import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict, Iterable

import numpy as np

from rl.distribution import Categorical, Choose
from rl.iterate import converged, iterate
from rl.markov_process import NonTerminal, State, Terminal
from rl.markov_decision_process import (MarkovDecisionProcess,
                                        FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy

import random 

from dataclasses import dataclass
from rl import dynamic_programming


from approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf


from approximate_dynamic_programming import QValueFunctionApprox
from td import PolicyFromQType, epsilon_greedy_action
from copy import copy

S = TypeVar('S')
A = TypeVar('A')



class TabularQValueFunctionApprox:
    def __init__(self):
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
    
    def update(self, k, tgt):
        alpha = 0.1
        self.values[k] = (1 - alpha) * self.values[k] + tgt * alpha
        self.counts[k] += 1
    
    def __call__(self, x_value) -> float:
        return self.values[x_value]


def merge(q_1: TabularQValueFunctionApprox, q_2: TabularQValueFunctionApprox):
    values = defaultdict(float)
    counts = defaultdict(int)
    for k in set(list(q_1.values.keys()) + list(q_2.values.keys())):
        normalizer = max(1, q_1.counts[k] + q_2.counts[k])
        values[k] = (q_1.values[k] * q_1.counts[k] + q_2.values[k] * q_2.counts[k]) / normalizer
        counts[k] = q_1.counts[k] + q_2.counts[k]
    retval = TabularQValueFunctionApprox()
    retval.counts = counts
    retval.values = values
    return retval


def double_q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float,
    max_episode_length: int
) -> Iterator[TabularQValueFunctionApprox]:
    ##### End Your Code HERE #########
    q_1: Mapping[S, A] = TabularQValueFunctionApprox()
    q_2: Mapping[S, A] = TabularQValueFunctionApprox()
    yield merge(q_1, q_2)
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            merged = merge(q_1, q_2)
            action: A = epsilon_greedy_action(merged, state,
                                              mdp.actions(state),
                                              ϵ=1e-1)
            next_state, reward = mdp.step(state, action).sample()
            if random.random() >= 0.5:
                if isinstance(next_state, NonTerminal):
                    next_action = max(mdp.actions(next_state),
                                      key = lambda a: q_1((next_state, a)))
                    next_return = q_2((next_state, next_action))
                else:
                    next_return = 0
                tgt = reward + γ * next_return
                q_1.update((state, action), tgt)
            else:
                if isinstance(next_state, NonTerminal):
                    next_action = max(mdp.actions(next_state),
                                      key = lambda a: q_2((next_state, a)))
                    next_return = q_1((next_state, next_action))
                else:
                    next_return = 0
                tgt = reward + γ * next_return
                q_2.update((state, action), tgt)
                
            yield merge(q_1, q_2)
            steps += 1
            state = next_state
    ##### End Your Code HERE #########
    
            
def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float,
    max_episode_length: int
) -> Iterator[TabularQValueFunctionApprox]:
    ##### Your Code HERE #########
    q: Mapping[S, A] = TabularQValueFunctionApprox()
    yield q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            action: A = epsilon_greedy_action(q,  state,
                                              mdp.actions(state),
                                              ϵ=1e-1)
            next_state, reward = mdp.step(state, action).sample()
            next_return: float = max(
                q((next_state, a))
                for a in mdp.actions(next_state)
            ) if isinstance(next_state, NonTerminal) else 0.
            tgt = reward + γ * next_return
            q.update((state, action), tgt)
            yield copy(q)
            steps += 1
            state = next_state
    ##### End Your Code HERE #########



@dataclass(frozen=True)
class P1State:
    ##### Your Code HERE #########
    state: str
        
    def sample_reward(self):
        if self.state == 'A':
            return 0
        elif self.state == 'B':
            return Gaussian(-0.1, 1).sample()
    
    ##### End Your Code HERE #########
    

class P1MDP(MarkovDecisionProcess[P1State, str]):

    def __init__(self, n):
        self.n = n
        
        
    def actions(self, state: NonTerminal[P1State]) -> Iterable[str]:
        ##### Your Code HERE #########
        if state.state.state == 'A':
            return ['L', 'R']
        elif state.state.state == 'B':
            return [str(i) for i in range(self.n)]
        ##### End Your Code HERE #########
    
    def step(
        self,
        state: NonTerminal[P1State],
        action: str
    ) -> Distribution[Tuple[State[P1State], float]]:
        ##### Your Code HERE #########
        if state.state.state == 'A':
            if action == 'L':
                return Constant((NonTerminal(P1State("B")), 0))
            elif action == 'R':
                return Constant((Terminal(P1State("T")), 0))
        elif state.state.state == 'B':
            return SampledDistribution(lambda: (Terminal(P1State("T")), state.state.sample_reward()))
    ##### End Your Code HERE #########