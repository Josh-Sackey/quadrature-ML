import numpy as np
from sklearn.preprocessing import StandardScaler
from adaptive.integrator import StateODE


class Predictor:
    def __call__(self, state):
        """
        Base Predictor call.

        Parameters
        ----------
        state : list[np.ndarray]

        Returns
        -------
        action: int
            the next step size should be step_sizes[action]
        """
        return 0


class PredictorQ(Predictor):
    def __init__(self, step_sizes, model, scaler):
        """
        Predictor using a neural network model.

        Parameters
        ----------
        step_sizes : list[float]
        model : tf.keras.Model
        scaler : StandardScaler
        """
        self.step_sizes = step_sizes
        self.scaler = scaler
        self.model = model

    def __call__(self, state):
        """
        Parameters
        ----------
        state : list[np.ndarray]

        Returns
        -------
        float
            the next step size
        """
        actions = self.model(self.scaler.transform([state[0]]))
        action = np.argmax(actions)
        return self.step_sizes[action]

    def get_actions(self, state):
        """
        Return the value of each possible action.

        Parameters
        ----------
        state : list[np.ndarray]

        Returns
        -------
        np.ndarray
        """
        return self.model(self.scaler.transform([state[0]])).numpy()
    
    def action_to_stepsize(self, action):
        """

        Parameters
        ----------
        action : int

        Returns
        -------
        float
            step_sizes[action]
        """
        return self.step_sizes[action]

    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
        actions : np.ndarray
        """
        return self.model.train_on_batch(self.scaler.transform(states), actions)


class PredictorConst(Predictor):
    def __init__(self, c):
        self.c = c

    def __call__(self, state):
        return self.c


class PredictorRandom(Predictor):
    def __init__(self, step_sizes):
        self.step_sizes = step_sizes

    def __call__(self, state):
        return np.random.choice(self.step_sizes)


class PredictorODE:
    def __call__(self, states):
        """
        Base Predictor call for ODE.

        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        float
        """
        return 0.1


class PredictorQODE(PredictorODE):
    def __init__(self, step_sizes, model, scaler, use_idx=False):
        """
        Predictor using a neural network model.

        Parameters
        ----------
        step_sizes : list[float]
        model : tf.keras.Model
        scaler : StandardScaler
        use_idx : bool, optional
            whether the step size is used in the state or a idx referring to it
            (e.g. if the step sizes are from different orders like [0.0001, 0.0002, 0.1], idx might work better)
        """
        self.use_idx = use_idx
        self.step_sizes = step_sizes
        self.scaler = scaler
        self.model = model

    def __call__(self, states, eps=0):
        """
        Parameters
        ----------
        states : list[StateODE]
        eps : float
            probability that a random action is chosen instead of the one with highest value

        Returns
        -------
        float
            step_sizes[action]
        """
        states = np.concatenate([state.flatten(self.use_idx) for state in states])
        actions = self.model(self.scaler.transform([states]))

        action = np.argmax(actions)
        rn = np.random.sample()

        if rn < 0.2 * eps:
            action = np.random.randint(len(self.step_sizes))
        elif rn < 0.6 * eps:
            action = min(action + 1, (len(self.step_sizes)) - 1)
        elif rn < eps:
            action = max(action - 1, 0)

        return self.step_sizes[action]

    def action_to_stepsize(self, action):
        """

        Parameters
        ----------
        action : int

        Returns
        -------
        float
            step_sizes[action]
        """
        return self.step_sizes[action]

    def get_actions(self, states):
        """
        Return the value of each possible action.

        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        np.ndarray
        """
        states = np.concatenate([state.flatten(self.use_idx) for state in states])
        return self.model(self.scaler.transform([states])).numpy()

    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
        actions : np.ndarray
        """
        return self.model.train_on_batch(self.scaler.transform(states), actions)


class PredictorConstODE(PredictorODE):
    def __init__(self, h):
        """

        Parameters
        ----------
        h : float
            constant step size

        """
        self.h = h

    def __call__(self, states):
        """
        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        float
        """
        return self.h


class MetaQODE(PredictorODE):
    def __init__(self, basis_learners, model, scaler, use_idx=False):
        """
        Metalearner using a neural network model.

        Parameters
        ----------
        basis_learners : list[PredictorODE]
        model : tf.keras.Model
        scaler : StandardScaler
        use_idx : bool, optional
            whether the step size is used in the state or a idx referring to it
            (e.g. if the step sizes are from different orders like [0.0001, 0.0002, 0.1], idx might work better)
        """
        self.use_idx = use_idx
        self.basis_learners = basis_learners
        self.scaler = scaler
        self.model = model

    def __call__(self, states, eps=0):
        """
        Parameters
        ----------
        states : list[StateODE]
        eps : float
            probability that a random action is chosen instead of the one with highest value

        Returns
        -------
        float
            step_size
        """
        states_np = np.concatenate([state.flatten(self.use_idx) for state in states])
        actions = self.model(self.scaler.transform([states_np]))

        action = np.argmax(actions)
        rn = np.random.sample()

        if rn < 0.2 * eps:
            action = np.random.randint(len(self.basis_learners))
        elif rn < 0.6 * eps:
            action = min(action + 1, (len(self.basis_learners)) - 1)
        elif rn < eps:
            action = max(action - 1, 0)

        bl = self.basis_learners[action]
        return bl(states[:1])

    def action_to_stepsize(self, action, states):
        """

        Parameters
        ----------
        action : int
        states : list[StateODE]

        Returns
        -------
        float
            step_sizes[action]
        """
        bl = self.basis_learners[action]
        return bl(states[:1])

    def get_actions(self, states):
        """
        Return the value of each possible action.

        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        np.ndarray
        """
        states = np.concatenate([state.flatten(self.use_idx) for state in states])
        return self.model(self.scaler.transform([states])).numpy()

    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
        actions : np.ndarray
        """
        return self.model.train_on_batch(self.scaler.transform(states), actions)
    
class PredictorQPDE(PredictorODE):
    def __init__(self, step_sizes, model, scaler, use_idx=False):
        """
        Predictor for PDEs using a neural network model.

        Parameters
        ----------
        step_sizes : list[float]
        model : tf.keras.Model
        scaler : StandardScaler
        use_idx : bool, optional
            Whether the step size is used in the state or an index referring to it.
        """
        self.use_idx = use_idx
        self.step_sizes = step_sizes
        self.scaler = scaler
        self.model = model

    def __call__(self, states, eps=0):
        """
        Parameters
        ----------
        states : list[PDEState]
            List of states containing PDE fields (e.g., spatial grids).
        eps : float
            Probability that a random action is chosen instead of the one with the highest value.

        Returns
        -------
        float
            step_sizes[action]
        """
        flattened_states = np.concatenate([state.flatten(self.use_idx) for state in states])
        actions = self.model(self.scaler.transform([flattened_states]))

        action = np.argmax(actions)
        rn = np.random.sample()

        if rn < 0.2 * eps:
            action = np.random.randint(len(self.step_sizes))
        elif rn < 0.6 * eps:
            action = min(action + 1, len(self.step_sizes) - 1)
        elif rn < eps:
            action = max(action - 1, 0)

        return self.step_sizes[action]

    def get_actions(self, states):
        """
        Return the value of each possible action.

        Parameters
        ----------
        states : list[PDEState]
            List of states containing PDE fields (e.g., spatial grids).

        Returns
        -------
        np.ndarray
        """
        flattened_states = np.concatenate([state.flatten(self.use_idx) for state in states])
        print("Shape of flattened_states before reshaping:", flattened_states.shape)
        flattened_states = flattened_states.reshape(1, -1)  # Ensure it's 2D
        print("Shape of flattened_states after reshaping:", flattened_states.shape)
        assert flattened_states.shape[1] == self.scaler.mean_.shape[0], \
            "Mismatch in number of features between scaler and input data"
        scaled_states = self.scaler.transform(flattened_states)
        print("Shape of scaled_states:", scaled_states.shape)
        return self.model(scaled_states).numpy()
    
    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
            Array of states transformed for PDE contexts.
        actions : np.ndarray
            Corresponding actions taken.

        Returns
        -------
        Loss or training feedback from the model.
        """
        transformed_states = self.scaler.transform(states)
        return self.model.train_on_batch(transformed_states, actions)

