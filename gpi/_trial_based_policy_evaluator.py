try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


def _make_hashable(s):
    """Convierte estados numpy array a bytes para usar como clave de dict."""
    if isinstance(s, np.ndarray):
        return s.tobytes()
    return s


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = gamma
        self.exploring_starts = exploring_starts
        self.max_trial_length = max_trial_length
        self.rng = random_state if random_state is not None else np.random.RandomState()
        self._v = {}
        self._q = {}

    def _generate_trial(self, policy):
        """
        Genera un trial usando SOLO la API pública de TrialInterface.
        Detecta estado terminal cuando get_actions_in_state devuelve lista vacía.
        """
        rows = []

        if self.exploring_starts:
            # Estado y acción iniciales aleatorios
            s, r = self.trial_interface.get_random_state()
            actions = self.trial_interface.get_actions_in_state(s)
            a0 = actions[self.rng.randint(0, len(actions))]
            rows.append([s, a0, r])
            s, r = self.trial_interface.exec_action(s, a0)
        else:
            s, r = self.trial_interface.draw_init_state()

        count = len(rows)
        actions = self.trial_interface.get_actions_in_state(s)
        while actions and count < self.max_trial_length:
            a = policy(s)
            rows.append([s, a, r])
            s, r = self.trial_interface.exec_action(s, a)
            actions = self.trial_interface.get_actions_in_state(s)
            count += 1

        rows.append([s, None, r])  # estado terminal
        return pd.DataFrame(rows, columns=["state", "action", "reward"])

    def step(self):
        policy = self.workspace.policy
        df_trial = self._generate_trial(policy)
        report = self.process_trial_for_policy(df_trial, policy)

        if self._v:
            self.workspace.replace_v(self._v)
        if self._q:
            self.workspace.replace_q(self._q)

        if isinstance(report, dict):
            report.setdefault("trial_length", len(df_trial))

        return report

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError

    @property
    def v(self):
        return self._v

    @property
    def q(self):
        return self._q

    @property
    def policy(self):
        return self.workspace.policy if self.workspace is not None else None