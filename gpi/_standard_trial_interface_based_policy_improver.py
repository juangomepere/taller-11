try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np

try:
    from ._trial_based_policy_evaluator import _make_hashable
except ImportError:
    from _trial_based_policy_evaluator import _make_hashable


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, trial_interface: TrialInterface, random_state: np.random.RandomState = None):
        super().__init__()
        self.trial_interface = trial_interface
        self.rng = random_state if random_state is not None else np.random.RandomState()
        self._policy_dict   = {}   # {key_s: a}
        self._actions_cache = {}   # {key_s: [actions]}  ← caché requerida por el test

    def _get_actions(self, s_raw):
        """Devuelve acciones válidas con caché para no llamar al interface repetidamente."""
        key = _make_hashable(s_raw)
        if key not in self._actions_cache:
            self._actions_cache[key] = self.trial_interface.get_actions_in_state(s_raw)
        return self._actions_cache[key]

    def step(self):
        q = self.workspace.q   # puede ser None al inicio
        has_improved    = False
        actions_changed = 0

        # Los estados a mejorar son los que ya tienen q-values en el workspace
        states_to_improve = set(self._policy_dict.keys())
        if q is not None:
            states_to_improve |= set(q.keys())

        # Guardar estados raw para poder llamar get_actions_in_state
        # (el workspace guarda raw states como claves en _q si los evaluadores
        #  usan _make_hashable → en LakeMDP las claves ya son tuplas hashables)
        for key_s in list(states_to_improve):
            # Obtener q-values para este estado
            if q is not None and key_s in q:
                q_s = q[key_s]   # {a: valor}
            else:
                q_s = {}

            # Para llamar get_actions_in_state necesitamos el estado raw.
            # En LakeMDP key_s == s_raw (tupla). En Connect4 key_s es bytes.
            # Usamos key_s directamente si es hashable tal como está.
            s_raw = key_s

            actions = self._get_actions(s_raw)
            if not actions:
                continue

            # Escoger la mejor acción (q desconocido → 0)
            best_a = max(actions, key=lambda a: q_s.get(a, 0.0))

            if self._policy_dict.get(key_s) != best_a:
                self._policy_dict[key_s] = best_a
                has_improved = True
                actions_changed += 1

        # Publicar política en el workspace
        snapshot = dict(self._policy_dict)

        def policy(s):
            key = _make_hashable(s)
            if key in snapshot:
                return snapshot[key]
            acts = self.trial_interface.get_actions_in_state(s)
            if acts:
                return acts[self.rng.randint(0, len(acts))]
            return None

        self.workspace.replace_policy(policy)

        return {
            "has_improved":    has_improved,
            "actions_changed": actions_changed,
        }