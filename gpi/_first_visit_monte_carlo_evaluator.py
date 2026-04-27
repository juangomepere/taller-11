import numpy as np
import pandas as pd
from mdp._trial_interface import TrialInterface

try:
    from ._trial_based_policy_evaluator import TrialBasedPolicyEvaluator, _make_hashable
except ImportError:
    from _trial_based_policy_evaluator import TrialBasedPolicyEvaluator, _make_hashable


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self._returns_sa = {}
        self._returns_s  = {}
        self.counts      = {}

    def process_trial_for_policy(self, df_trial, policy):
        """
        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :return: returns a depth-2 dictionary that contains the *change* in the q-values
                 (np.inf if a q-value was not available before)
        """
        n = len(df_trial)
        G        = 0.0
        first_sa = {}
        first_s  = {}

        for t in range(n - 1, -1, -1):
            s_raw = df_trial.iloc[t]["state"]
            a     = df_trial.iloc[t]["action"]
            r     = float(df_trial.iloc[t]["reward"])
            G     = r + self.gamma * G

            key_s = _make_hashable(s_raw)
            valid = a is not None and not (isinstance(a, float) and np.isnan(a))

            if valid:
                first_sa[(key_s, a)] = (G, s_raw)
            first_s[key_s] = (G, s_raw)

        changes = {}

        for (key_s, a), (G, s_raw) in first_sa.items():
            self._returns_sa.setdefault((key_s, a), []).append(G)
            q_new = float(np.mean(self._returns_sa[(key_s, a)]))
            q_old = self._q.get(key_s, {}).get(a, None)

            changes.setdefault(key_s, {})[a] = np.inf if q_old is None else (q_new - q_old)
            self._q.setdefault(key_s, {})[a] = q_new
            self.counts[(key_s, a)] = self.counts.get((key_s, a), 0) + 1

        for key_s, (G, s_raw) in first_s.items():
            self._returns_s.setdefault(key_s, []).append(G)
            self._v[key_s] = float(np.mean(self._returns_s[key_s]))

        return {
            "trial_length": n,
            "q_changes":    changes,
            "states_seen":  len(first_s),
        }