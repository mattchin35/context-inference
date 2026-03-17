import numpy as np
from scipy.special import expit, logit

N_ACTIONS = 2
RIGHT_IX = 0
LEFT_IX = 1
eps = np.finfo(float).eps

states = ["right", "left"]
side_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]


def get_action_ix(action: int) -> int:
    # right=0, left=1: map right to -1, left to +1
    assert action in [0, 1], "action must be 0 or 1"
    return action * 2 - 1


def qlearning_relative_value(
    actions, rewards, learning_rate=0.1, n_actions=N_ACTIONS, give_reward=None
):
    """
    Minimal Q-learning relative value feature.
    Returns trial-wise pre-update value: Q_left - Q_right.
    """
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    _validate_lengths(actions, rewards)

    skip_trials = _normalize_skip_trials(give_reward, actions.shape[0])

    Q = np.zeros(n_actions)
    if skip_trials is None:
        relative_value = np.zeros(actions.shape[0], dtype=float)
    else:
        relative_value = np.full(actions.shape[0], None, dtype=object)

    for i, (action, reward) in enumerate(zip(actions, rewards)):
        if skip_trials is not None and skip_trials[i]:
            continue

        relative_value[i] = Q[LEFT_IX] - Q[RIGHT_IX]
        action = _parse_action(action, n_actions)
        reward = _parse_reward(reward)
        if action is None or reward is None:
            continue
        Q[action] += learning_rate * (reward - Q[action])

    return relative_value


def forgetting_qlearning_relative_value(
    actions,
    rewards,
    decay=0.9,
    reward_update_rate=None,
    tanh_relative_value=False,
    tanh_scale=1.0,
    n_actions=N_ACTIONS,
    give_reward=None,
):
    """
    Minimal forgetting Q-learning relative value feature.
    Returns trial-wise pre-update value: Q_left - Q_right.

    Update rule:
        Q *= decay
        Q[action] += reward_update_rate * reward

    If reward_update_rate is None, defaults to (1 - decay), matching the original behavior.
    Optional output transform:
        If tanh_relative_value is True, return tanh(tanh_scale * (Q_left - Q_right)).
        This keeps values in [-1, 1] and is most useful when reward_update_rate differs
        from (1 - decay).
    """
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    _validate_lengths(actions, rewards)
    if not 0 <= decay <= 1:
        raise ValueError("decay must be in [0, 1].")
    if reward_update_rate is None:
        reward_update_rate = 1 - decay
    if not 0 <= reward_update_rate <= 1:
        raise ValueError("reward_update_rate must be in [0, 1].")
    if not isinstance(tanh_relative_value, (bool, np.bool_)):
        raise ValueError("tanh_relative_value must be a boolean.")
    if tanh_relative_value:
        try:
            tanh_scale = float(tanh_scale)
        except (TypeError, ValueError):
            raise ValueError(
                "tanh_scale must be a positive float when tanh_relative_value is True."
            )
        if tanh_scale <= 0:
            raise ValueError("tanh_scale must be > 0 when tanh_relative_value is True.")

    skip_trials = _normalize_skip_trials(give_reward, actions.shape[0])

    Q = np.zeros(n_actions)
    if skip_trials is None:
        relative_value = np.zeros(actions.shape[0], dtype=float)
    else:
        relative_value = np.full(actions.shape[0], None, dtype=object)

    for i, (action, reward) in enumerate(zip(actions, rewards)):
        if skip_trials is not None and skip_trials[i]:
            continue

        relative_value_raw = Q[LEFT_IX] - Q[RIGHT_IX]
        if tanh_relative_value:
            relative_value[i] = np.tanh(tanh_scale * relative_value_raw)
        else:
            relative_value[i] = relative_value_raw
        action = _parse_action(action, n_actions)
        reward = _parse_reward(reward)
        if action is None or reward is None:
            continue

        # Vertechi et al. Neuron 2020 has reward update as (1 - decay) * reward
        Q *= decay
        Q[action] += reward_update_rate * reward

    return relative_value


def hmm_relative_value(
    actions,
    rewards,
    state_transition_prob=0.2,
    active_reward_probability=0.8,
    inactive_reward_probability=0.0,
    correct_reward_size=1.0,
    incorrect_reward_size=0.0,
    value_mode="expected_reward",
    tanh_scale=1,
    give_reward=None,
):
    """
    Minimal HMM relative value feature.
    Returns trial-wise pre-update value from belief state.

    value_mode:
        - "expected_reward": E[r_left] - E[r_right]
        - "bayesian_log_odds": log(p_left / p_right)
    """
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    _validate_lengths(actions, rewards)
    if not 0 <= state_transition_prob <= 1:
        raise ValueError("state_transition_prob must be in [0, 1].")
    if value_mode not in ("expected_reward", "bayesian_log_odds"):
        raise ValueError("value_mode must be 'expected_reward' or 'bayesian_log_odds'.")

    skip_trials = _normalize_skip_trials(give_reward, actions.shape[0])

    prior = np.ones(2) / 2
    if skip_trials is None:
        relative_value = np.zeros(actions.shape[0], dtype=float)
    else:
        relative_value = np.full(actions.shape[0], None, dtype=object)
    transition_matrix = np.ones((2, 2)) * state_transition_prob
    np.fill_diagonal(transition_matrix, 1 - state_transition_prob)

    for i, (action, reward) in enumerate(zip(actions, rewards)):
        if skip_trials is not None and skip_trials[i]:
            continue

        if value_mode == "expected_reward":
            expected_rew_left = (
                active_reward_probability * prior[LEFT_IX]
                + inactive_reward_probability * prior[RIGHT_IX]
            )
            expected_rew_right = (
                active_reward_probability * prior[RIGHT_IX]
                + inactive_reward_probability * prior[LEFT_IX]
            )
            relative_value[i] = expected_rew_left - expected_rew_right
        else:
            log_odds = np.log((prior[LEFT_IX] + eps) / (prior[RIGHT_IX] + eps))
            relative_value[i] = np.tanh(log_odds * tanh_scale)

        action = _parse_action(action, N_ACTIONS)
        reward = _parse_reward(reward)
        if action is None or reward is None:
            continue

        p_reward_delivery = _reward_delivery_probability(
            action=action,
            active_reward_probability=active_reward_probability,
            inactive_reward_probability=inactive_reward_probability,
        )
        p_reward_size = _nonzero_reward_size_probability(
            reward=reward,
            action=action,
            correct_reward_size=correct_reward_size,
            incorrect_reward_size=incorrect_reward_size,
        )

        nonzero_reward_indicator = int(reward > 0)
        p_no_reward = 1 - p_reward_delivery
        p_reward = (
            nonzero_reward_indicator * p_reward_delivery * p_reward_size
            + (1 - nonzero_reward_indicator) * p_no_reward
        )

        p_outcome = p_reward * prior
        p_outcome /= np.sum(p_outcome) + eps

        posterior = np.dot(transition_matrix.T, p_outcome)
        posterior /= np.sum(posterior) + eps
        prior = posterior

    return relative_value


def hmm_relative_value_reward_decay(
    actions,
    rewards,
    state_transition_prob=0.2,
    active_reward_probability=0.8,
    inactive_reward_probability=0.0,
    correct_reward_size=1.0,
    incorrect_reward_size=0.0,
    lambda_decay=0.2,
    value_mode="bayesian_log_odds",
    tanh_scale=1.0,
    give_reward=None,
):
    """
    HMM relative value variant with asymmetric outcome updates:
      - reward=1: Bayesian update
      - reward=0: passive decay in log-odds space

    Output is signed belief only:
        tanh(tanh_scale * logit(p_left_prior))

    Notes:
      - B_prior and transition_uncertainty are computed internally each trial
        but not returned.
      - For readability/comparison with hmm_relative_value, trial loop order is:
        1) compute/store output from current prior
        2) outcome update (reward vs omission)
        3) hazard/transition update for next trial
    """
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    _validate_lengths(actions, rewards)
    if not 0 <= state_transition_prob <= 1:
        raise ValueError("state_transition_prob must be in [0, 1].")
    if not 0 <= lambda_decay <= 1:
        raise ValueError("lambda_decay must be in [0, 1].")
    if value_mode != "bayesian_log_odds":
        raise ValueError("value_mode must be 'bayesian_log_odds'.")
    try:
        tanh_scale = float(tanh_scale)
    except (TypeError, ValueError):
        raise ValueError("tanh_scale must be a positive float.")
    if tanh_scale <= 0:
        raise ValueError("tanh_scale must be > 0.")

    skip_trials = _normalize_skip_trials(give_reward, actions.shape[0])

    # Belief vector order: [p_right, p_left]
    prior = np.ones(2) / 2
    if skip_trials is None:
        relative_value = np.zeros(actions.shape[0], dtype=float)
    else:
        relative_value = np.full(actions.shape[0], None, dtype=object)

    transition_matrix = np.ones((2, 2)) * state_transition_prob
    np.fill_diagonal(transition_matrix, 1 - state_transition_prob)

    for i, (action, reward) in enumerate(zip(actions, rewards)):
        if skip_trials is not None and skip_trials[i]:
            continue

        # 1) Start-of-trial regressors from current prior belief
        p_left_prior = np.clip(prior[LEFT_IX], eps, 1 - eps)
        B_prior = logit(p_left_prior)
        relative_value[i] = np.tanh(tanh_scale * B_prior)  # signed_belief
        transition_uncertainty = 1 - np.abs(
            2 * p_left_prior - 1
        )  # computed but not returned
        _ = transition_uncertainty

        action = _parse_action(action, N_ACTIONS)
        reward = _parse_reward(reward)
        if action is None or reward is None:
            continue

        # 2) Outcome update
        if np.isclose(reward, 1.0):
            # Bayesian reward update on reward delivery
            p_reward_delivery = _reward_delivery_probability(
                action=action,
                active_reward_probability=active_reward_probability,
                inactive_reward_probability=inactive_reward_probability,
            )
            p_reward_size = _nonzero_reward_size_probability(
                reward=reward,
                action=action,
                correct_reward_size=correct_reward_size,
                incorrect_reward_size=incorrect_reward_size,
            )
            likelihood = p_reward_delivery * p_reward_size
            posterior = likelihood * prior
            posterior /= np.sum(posterior) + eps
        elif np.isclose(reward, 0.0):
            # Passive decay in log-odds space
            B_post = (1 - lambda_decay) * B_prior
            p_left_post = expit(B_post)
            posterior = np.array([1 - p_left_post, p_left_post], dtype=float)
        else:
            # Fallback for non-binary rewards: treat positive as reward, non-positive as omission.
            if reward > 0:
                p_reward_delivery = _reward_delivery_probability(
                    action=action,
                    active_reward_probability=active_reward_probability,
                    inactive_reward_probability=inactive_reward_probability,
                )
                p_reward_size = _nonzero_reward_size_probability(
                    reward=reward,
                    action=action,
                    correct_reward_size=correct_reward_size,
                    incorrect_reward_size=incorrect_reward_size,
                )
                likelihood = p_reward_delivery * p_reward_size
                posterior = likelihood * prior
                posterior /= np.sum(posterior) + eps
            else:
                B_post = (1 - lambda_decay) * B_prior
                p_left_post = expit(B_post)
                posterior = np.array([1 - p_left_post, p_left_post], dtype=float)

        # 3) Hazard/transition update for next trial
        prior = np.dot(transition_matrix.T, posterior)
        prior /= np.sum(prior) + eps

    return relative_value


def _validate_lengths(actions, rewards):
    if actions.shape[0] != rewards.shape[0]:
        raise ValueError("actions and rewards must have the same length.")


def _normalize_skip_trials(give_reward, n_trials):
    if give_reward is None:
        return None

    give_reward = np.asarray(give_reward)
    if give_reward.shape[0] != n_trials:
        raise ValueError(
            "give_reward must have the same length as actions and rewards."
        )

    return np.array([_should_skip_trial(flag) for flag in give_reward], dtype=bool)


def _should_skip_trial(flag):
    if flag is None:
        return False

    try:
        if np.isnan(flag):
            return False
    except TypeError:
        pass

    if isinstance(flag, str):
        parsed = flag.strip().lower()
        if parsed in {"1", "true", "t", "yes", "y"}:
            return True
        if parsed in {"0", "false", "f", "no", "n", "none", ""}:
            return False
        try:
            return bool(int(float(parsed)))
        except (TypeError, ValueError):
            return False

    return bool(flag)


def _parse_action(action, n_actions):
    try:
        if np.isnan(action):
            return None
    except TypeError:
        pass

    try:
        action = int(action)
    except (TypeError, ValueError):
        return None

    if action < 0 or action >= n_actions:
        return None

    return action


def _parse_reward(reward):
    try:
        if np.isnan(reward):
            return None
    except TypeError:
        pass

    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _reward_delivery_probability(
    action, active_reward_probability, inactive_reward_probability
):
    p = np.zeros(2)
    if action == RIGHT_IX:
        p[RIGHT_IX] = active_reward_probability
        p[LEFT_IX] = inactive_reward_probability
    elif action == LEFT_IX:
        p[RIGHT_IX] = inactive_reward_probability
        p[LEFT_IX] = active_reward_probability
    return p


def _nonzero_reward_size_probability(
    reward, action, correct_reward_size, incorrect_reward_size
):
    p = np.zeros(2)
    if action == RIGHT_IX:
        p[RIGHT_IX] = float(reward == correct_reward_size)
        p[LEFT_IX] = float(reward == incorrect_reward_size)
    elif action == LEFT_IX:
        p[RIGHT_IX] = float(reward == incorrect_reward_size)
        p[LEFT_IX] = float(reward == correct_reward_size)
    return p


def relative_omissions_index(R_omissions, L_omissions, epsilon=1e-8):
    """
    Compute relative omissions index in [-1, 1].

    Parameters
    ----------
    R_omissions : float or array-like
        Consecutive unrewarded trials on the right side.
    L_omissions : float or array-like
        Consecutive unrewarded trials on the left side.
    epsilon : float
        Small constant to prevent division by zero.

    Returns
    -------
    O : float or np.ndarray
        Relative omissions index in [-1, 1].
        +1 → only left accumulating omissions
        -1 → only right accumulating omissions
         0 → equal omissions
    """

    R = np.asarray(R_omissions, dtype=float)
    L = np.asarray(L_omissions, dtype=float)

    # Left-minus-right convention so left is positive and right is negative.
    return (L - R) / (R + L + epsilon)


def signed_omission_regressor(loss_streak, lam, choice_side):
    """
    Compute signed omission regressor using exponential saturating doubt. This is for use with a combined omissions
    count, reflecting overall doubt rather than side-specific doubt. The sign is determined by the current choice.

    Parameters
    ----------
    loss_streak : float or array-like
        Consecutive unrewarded trial count (>= 0).
    lam : float
        Saturation rate parameter (lambda > 0).
    choice_side : str or int or array-like
        Current choice.
        Accepts:
            - 'right' / 'left'
            - 0 (right) / 1 (left)
            - -1 (right) / +1 (left)

    Returns
    -------
    O : float or np.ndarray
        Signed omission regressor in [-1, 1].
    """

    # Convert to numpy array for vectorization
    D = np.asarray(loss_streak)

    # Unsigned doubt in [0, 1)
    H = 1 - np.exp(-lam * D)

    # Signed choice convention: left=+1, right=-1.
    sign = _choice_to_signed(choice_side)

    return sign * H


def relative_doubt_index(R_omissions, L_omissions, lam):
    """
    Compute relative doubt index using exponential saturating omission
    for both sides, returning a value in [-1, 1].

    Parameters
    ----------
    R_omissions : float or array-like
        Consecutive unrewarded trials on the right side.
    L_omissions : float or array-like
        Consecutive unrewarded trials on the left side.
    lam : float
        Exponential saturation parameter (lambda > 0).

    Returns
    -------
    D : float or np.ndarray
        Relative doubt index in [-1, 1].
        +1 → strong doubt on left only
        -1 → strong doubt on right only
         0 → equal doubt
    """

    R = np.asarray(R_omissions, dtype=float)
    L = np.asarray(L_omissions, dtype=float)

    # Exponential saturating doubt per side
    H_R = 1 - np.exp(-lam * R)
    H_L = 1 - np.exp(-lam * L)

    # Left-minus-right convention so left is positive and right is negative.
    return H_L - H_R


def perseveration_regressor(choices, decay=0.25):
    """
    Compute exponentially weighted perseveration regressor.

    Parameters
    ----------
    choices : array-like of shape (T,)
        Choices coded as either:
            - task-coded: 0 = right, 1 = left
            - signed: -1 = right, +1 = left
            - strings: 'right' / 'left'
    decay : float
        Exponential decay constant (lambda). Default = 0.25.

    Returns
    -------
    pers : np.ndarray of shape (T,)
        Perseveration regressor in [-1, 1].
        pers[t] depends only on trials < t.
    """
    choices = np.asarray(choices)
    T = len(choices)

    # Signed choice convention: left=+1, right=-1.
    y = _choice_to_signed(choices)

    pers = np.zeros(T)
    num = 0.0  # weighted numerator
    den = 0.0  # weighted normalization
    alpha = np.exp(-decay)

    for t in range(1, T):
        num = alpha * num + alpha * y[t - 1]
        den = alpha * den + alpha
        pers[t] = num / den if den > 0 else 0.0

    return pers


def perseveration_regressor_vectorized(choices, decay=0.25):
    """
    Vectorized exponentially weighted perseveration regressor.

    Parameters
    ----------
    choices : array-like (T,)
        Choices coded as either:
            - task-coded: 0 = right, 1 = left
            - signed: -1 = right, +1 = left
            - strings: 'right' / 'left'
    decay : float
        Exponential decay parameter (lambda)

    Returns
    -------
    pers : np.ndarray (T,)
        Perseveration regressor in [-1, 1]
        pers[t] depends only on trials < t
    """
    choices = np.asarray(choices)
    T = len(choices)

    # Signed choice convention: left=+1, right=-1.
    y = _choice_to_signed(choices)

    alpha = np.exp(-decay)

    # Exponential multipliers
    powers = alpha ** np.arange(T)

    # Weighted cumulative numerator
    weighted_y = y * powers
    cumsum_num = np.cumsum(weighted_y)

    # Denominator cumulative weights
    cumsum_den = np.cumsum(powers)

    # Shift to ensure pers[t] only uses trials < t
    num = np.zeros(T)
    den = np.zeros(T)

    num[1:] = cumsum_num[:-1]
    den[1:] = cumsum_den[:-1]

    pers = np.zeros(T)
    valid = den > 0
    pers[valid] = num[valid] / den[valid]

    return pers


def _choice_to_signed(choice_side):
    """
    Map choices to sign convention used by relative-value features:
    left -> +1, right -> -1.
    """
    choice_arr = np.asarray(choice_side)
    if choice_arr.ndim == 0:
        return _choice_token_to_signed(choice_arr.item())

    signed = np.empty(choice_arr.shape, dtype=float)
    for idx, token in np.ndenumerate(choice_arr):
        signed[idx] = _choice_token_to_signed(token)
    return signed


def _choice_token_to_signed(token):
    if isinstance(token, str):
        parsed = token.strip().lower()
        if parsed == "left":
            return 1.0
        if parsed == "right":
            return -1.0
        try:
            token = float(parsed)
        except ValueError as exc:
            raise ValueError(
                "choice values must be left/right, 0/1, or -1/+1."
            ) from exc

    try:
        if np.isnan(token):
            raise ValueError("choice values cannot be NaN.")
    except TypeError:
        pass

    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("choice values must be left/right, 0/1, or -1/+1.") from exc

    if value in (1.0, float(LEFT_IX)):
        return 1.0
    if value in (-1.0, float(RIGHT_IX)):
        return -1.0

    raise ValueError("choice values must be left/right, 0/1, or -1/+1.")
