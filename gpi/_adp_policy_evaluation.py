from mdp._trial_interface import TrialInterface
import numpy as np
from collections import defaultdict

from policy_evaluation._linear import LinearSystemEvaluator
from mdp._base import ClosedFormMDP

try:
    from ._trial_based_policy_evaluator import TrialBasedPolicyEvaluator, _make_hashable
except ImportError:
    from _trial_based_policy_evaluator import TrialBasedPolicyEvaluator, _make_hashable


def _has_action(action):
    return action is not None and not (
        isinstance(action, (float, np.floating)) and np.isnan(action)
    )


class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
        precision_for_transition_probability_estimates=4,
        update_interval: int = 10,
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self.precision        = precision_for_transition_probability_estimates
        self.update_interval  = update_interval
        self.steps_taken      = 0
        self.linear_evaluator = None

        # ← lista ordenada de raw states (orden de primera aparición)
        self.state_vector = []

        # counts[s_raw][a][s'_raw] = int  (raw states como clave)
        self.counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        self._reward_obs    = defaultdict(list)   # {key_s: [r, ...]}
        self._seen_keys     = set()               # para no duplicar en state_vector
        self._known_actions = set()

    def _register_state(self, s_raw):
        """Añade s_raw a state_vector la primera vez que se ve."""
        key_s = _make_hashable(s_raw)
        if key_s not in self._seen_keys:
            self._seen_keys.add(key_s)
            self.state_vector.append(s_raw)
        return key_s

    def _synchronize_knowledge_about_states_and_actions(self, s, a, r):
        key_s = self._register_state(s)
        self._reward_obs[key_s].append(r)
        if _has_action(a):
            self._known_actions.add(a)

    def get_believed_probs(self) -> dict:
        """
        :return: the 3-dim tensor where P[s][a][s'] is the *estimate* of
                 P(s'|s,a) based on the current knowledge
        """
        believed = {}
        for s, a_dict in self.counts.items():
            believed[s] = {}
            for a, sp_dict in a_dict.items():
                total = sum(sp_dict.values())
                if total == 0:
                    continue
                believed[s][a] = {
                    sp: round(cnt / total, self.precision)
                    for sp, cnt in sp_dict.items()
                    if cnt > 0
                }
        return believed

    def _rebuild_and_evaluate(self, policy):
        raw_states = self.state_vector          # orden de descubrimiento
        actions    = sorted(self._known_actions, key=str)

        if not raw_states or not actions:
            return

        n_s   = len(raw_states)
        n_a   = len(actions)
        s_idx = {_make_hashable(s): i for i, s in enumerate(raw_states)}
        a_idx = {a: j for j, a in enumerate(actions)}

        prob_matrix = np.zeros((n_s, n_a, n_s))

        for s_raw, a_dict in self.counts.items():
            key_s = _make_hashable(s_raw)
            if key_s not in s_idx:
                continue
            i = s_idx[key_s]
            for a, sp_dict in a_dict.items():
                if a not in a_idx:
                    continue
                j = a_idx[a]
                total = sum(sp_dict.values())
                if total == 0:
                    continue
                for sp_raw, cnt in sp_dict.items():
                    key_sp = _make_hashable(sp_raw)
                    if key_sp in s_idx:
                        k = s_idx[key_sp]
                        prob_matrix[i, j, k] = cnt / total

        rewards = np.array([
            float(np.mean(self._reward_obs[_make_hashable(s)])) if self._reward_obs[_make_hashable(s)] else 0.0
            for s in raw_states
        ])

        mdp_est = ClosedFormMDP(raw_states, actions, prob_matrix, rewards)

        try:
            self.linear_evaluator = LinearSystemEvaluator(mdp_est, self.gamma)
            self.linear_evaluator.reset(policy)
            self._v = {_make_hashable(s): v for s, v in self.linear_evaluator.v.items()}
            self._q = {_make_hashable(s): qa for s, qa in self.linear_evaluator.q.items()}
        except Exception:
            pass

    def process_trial_for_policy(self, df_trial, policy):
        """
        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """
        n = len(df_trial)

        for t in range(n):
            s_raw = df_trial.iloc[t]["state"]
            a     = df_trial.iloc[t]["action"]
            r     = float(df_trial.iloc[t]["reward"])

            self._synchronize_knowledge_about_states_and_actions(s_raw, a, r)

            if _has_action(a) and t + 1 < n:
                sp_raw = df_trial.iloc[t + 1]["state"]
                self._register_state(sp_raw)
                self.counts[s_raw][a][sp_raw] += 1

        self.steps_taken += 1
        if self.steps_taken % self.update_interval == 0:
            self._rebuild_and_evaluate(policy)

        return {
            "trial_length":  n,
            "known_states":  len(self.state_vector),
            "known_actions": len(self._known_actions),
            "steps_taken":   self.steps_taken,
        }