from __future__ import annotations

import time
import numpy as np

from connect4.policy import Policy
from connect4.trial_interface import Connect4TrialInterface


def learn_policy(
    trial_interface: Connect4TrialInterface,
    timeout: int | None = None,
) -> Policy:
    from gpi._base import GeneralPolicyIteration
    from gpi._first_visit_monte_carlo_evaluator import FirstVisitMonteCarloEvaluator
    from gpi._standard_trial_interface_based_policy_improver import (
        StandardTrialInterfaceBasedPolicyImprover,
    )

    rng   = np.random.RandomState(42)
    gamma = 0.99

    evaluator = FirstVisitMonteCarloEvaluator(
        trial_interface=trial_interface,
        gamma=gamma,
        exploring_starts=True,
        random_state=rng,
    )
    improver = StandardTrialInterfaceBasedPolicyImprover(
        trial_interface=trial_interface,
        random_state=rng,
    )

    gpi = GeneralPolicyIteration(gamma=gamma, components=[evaluator, improver])

    def random_policy(s):
        acts = trial_interface.get_actions_in_state(s)
        return acts[rng.randint(0, len(acts))] if acts else None

    gpi.workspace.replace_policy(random_policy)

    start    = time.time()
    max_iter = 200_000
    for _ in range(max_iter):
        if timeout is not None and (time.time() - start) >= timeout:
            break
        gpi.step()

    learned = gpi.policy

    class LearnedPolicy(Policy):
        def act(self, state):
            a = learned(state)
            if a is None:
                acts = trial_interface.get_actions_in_state(state)
                return acts[rng.randint(0, len(acts))] if acts else 0
            return a

    return LearnedPolicy()