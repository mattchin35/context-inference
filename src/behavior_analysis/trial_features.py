import numpy as np
from scipy.special import expit, logit
from enum import IntEnum

N_ACTIONS = 2
eps = np.finfo(float).eps


class ChoiceSide(IntEnum):
    RIGHT = 0
    LEFT = 1


SIGNED_RIGHT = -1.0
SIGNED_LEFT = 1.0
RIGHT_CHOICE_LABELS = {"right"}
LEFT_CHOICE_LABELS = {"left"}
NO_CHOICE_ACTION_LABELS = {"none", "no_choice", ""}


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
    _validate_two_action_task(n_actions)

    skip_trials = make_skip_trial_mask(give_reward=give_reward, actions=actions)

    Q = np.zeros(n_actions)
    if skip_trials is None:
        relative_value = np.zeros(actions.shape[0], dtype=float)
    else:
        relative_value = np.full(actions.shape[0], None, dtype=object)

    for i, (action, reward) in enumerate(zip(actions, rewards)):
        if skip_trials is not None and skip_trials[i]:
            continue

        relative_value[i] = Q[ChoiceSide.LEFT] - Q[ChoiceSide.RIGHT]
        action = _parse_choice_side(action, n_actions)
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
    _validate_two_action_task(n_actions)

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

    skip_trials = make_skip_trial_mask(give_reward=give_reward, actions=actions)

    Q = np.zeros(n_actions)
    if skip_trials is None:
        relative_value = np.zeros(actions.shape[0], dtype=float)
    else:
        relative_value = np.full(actions.shape[0], None, dtype=object)

    for i, (action, reward) in enumerate(zip(actions, rewards)):
        if skip_trials is not None and skip_trials[i]:
            continue

        relative_value_raw = Q[ChoiceSide.LEFT] - Q[ChoiceSide.RIGHT]
        if tanh_relative_value:
            relative_value[i] = np.tanh(tanh_scale * relative_value_raw)
        else:
            relative_value[i] = relative_value_raw
        action = _parse_choice_side(action, n_actions)
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

    skip_trials = make_skip_trial_mask(give_reward=give_reward, actions=actions)

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

        # Start-of-trial value from current prior
        if value_mode == "expected_reward":
            expected_rew_left = (
                active_reward_probability * prior[ChoiceSide.LEFT]
                + inactive_reward_probability * prior[ChoiceSide.RIGHT]
            )
            expected_rew_right = (
                active_reward_probability * prior[ChoiceSide.RIGHT]
                + inactive_reward_probability * prior[ChoiceSide.LEFT]
            )
            relative_value[i] = expected_rew_left - expected_rew_right
        else:
            log_odds = np.log((prior[ChoiceSide.LEFT] + eps) / (prior[ChoiceSide.RIGHT] + eps))
            relative_value[i] = np.tanh(log_odds * tanh_scale)

        action = _parse_choice_side(action, N_ACTIONS)
        reward = _parse_reward(reward)
        if action is None or reward is None:
            continue

        # Outcome likelihood update
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

        # Hazard/transition update
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

    skip_trials = make_skip_trial_mask(give_reward=give_reward, actions=actions)

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
        p_left_prior = np.clip(prior[ChoiceSide.LEFT], eps, 1 - eps)
        B_prior = logit(p_left_prior)
        relative_value[i] = np.tanh(tanh_scale * B_prior)  # signed_belief

        action = _parse_choice_side(action, N_ACTIONS)
        reward = _parse_reward(reward)
        if action is None or reward is None:
            continue

        # 2) Outcome update
        if np.isclose(reward, 1.0):
            posterior = _bayesian_reward_posterior(
                prior=prior,
                action=action,
                reward=reward,
                active_reward_probability=active_reward_probability,
                inactive_reward_probability=inactive_reward_probability,
                correct_reward_size=correct_reward_size,
                incorrect_reward_size=incorrect_reward_size,
            )
        elif np.isclose(reward, 0.0):
            posterior = _decayed_log_odds_posterior(
                B_prior=B_prior,
                lambda_decay=lambda_decay,
            )
        else:
            # Fallback for non-binary rewards: treat positive as reward, non-positive as omission.
            if reward > 0:
                posterior = _bayesian_reward_posterior(
                    prior=prior,
                    action=action,
                    reward=reward,
                    active_reward_probability=active_reward_probability,
                    inactive_reward_probability=inactive_reward_probability,
                    correct_reward_size=correct_reward_size,
                    incorrect_reward_size=incorrect_reward_size,
                )
            else:
                posterior = _decayed_log_odds_posterior(
                    B_prior=B_prior,
                    lambda_decay=lambda_decay,
                )

        # 3) Hazard/transition update for next trial
        prior = np.dot(transition_matrix.T, posterior)
        prior /= np.sum(prior) + eps

    return relative_value


def _bayesian_reward_posterior(
    prior,
    action,
    reward,
    active_reward_probability,
    inactive_reward_probability,
    correct_reward_size,
    incorrect_reward_size,
):
    """Return normalized HMM reward posterior.

    Parameters
    ----------
    prior : np.ndarray, shape (2,)
        Prior state probabilities ordered as [right, left].
    action : ChoiceSide
        Parsed choice side for the current trial.
    reward : float
        Observed reward magnitude for the current trial.
    active_reward_probability : float
        Reward-delivery probability for the active side.
    inactive_reward_probability : float
        Reward-delivery probability for the inactive side.
    correct_reward_size : float
        Reward magnitude expected for rewarded correct choices.
    incorrect_reward_size : float
        Reward magnitude expected for unrewarded or incorrect choices.

    Returns
    -------
    np.ndarray, shape (2,)
        Posterior state probabilities ordered as [right, left].
    """
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
    posterior = p_reward_delivery * p_reward_size * prior
    posterior /= np.sum(posterior) + eps
    return posterior


def _decayed_log_odds_posterior(B_prior, lambda_decay):
    """Return posterior after passive omission decay in log-odds space.

    Parameters
    ----------
    B_prior : float
        Prior left-vs-right log odds for the current trial.
    lambda_decay : float
        Passive decay fraction in [0, 1].

    Returns
    -------
    np.ndarray, shape (2,)
        Posterior state probabilities ordered as [right, left].
    """
    B_post = (1 - lambda_decay) * B_prior
    p_left_post = expit(B_post)
    return np.array([1 - p_left_post, p_left_post], dtype=float)


def _validate_lengths(actions, rewards):
    if actions.shape[0] != rewards.shape[0]:
        raise ValueError("actions and rewards must have the same length.")


def _validate_two_action_task(n_actions):
    if n_actions != N_ACTIONS:
        raise ValueError(
            "trial_features currently supports only two actions: 0=right, 1=left."
        )


def make_skip_trial_mask(give_reward, actions):
    """Return rows to skip for manual-reward or no-choice trials.

    Parameters
    ----------
    give_reward : array-like or None
        Experimenter-reward flags with shape `(n_trials,)`, or None when the
        column is absent.
    actions : array-like
        Trial actions with shape `(n_trials,)`.

    Returns
    -------
    np.ndarray or None
        Boolean mask with shape `(n_trials,)`, or None when no rows should be
        skipped.
    """
    actions = np.asarray(actions)
    no_choice_mask = np.array([_is_no_choice_action(action) for action in actions], dtype=bool)

    if give_reward is None:
        return no_choice_mask if no_choice_mask.any() else None

    give_reward = np.asarray(give_reward)
    if give_reward.shape[0] != actions.shape[0]:
        raise ValueError(
            "give_reward must have the same length as actions and rewards."
        )

    skip_mask = np.array([_should_skip_give_reward(flag) for flag in give_reward], dtype=bool)
    skip_mask = skip_mask | no_choice_mask
    return skip_mask if skip_mask.any() else None


def _should_skip_give_reward(flag):
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


def _is_no_choice_action(action):
    if action is None:
        return True

    if isinstance(action, str):
        return action.strip().lower() in NO_CHOICE_ACTION_LABELS

    try:
        return bool(np.isnan(action))
    except TypeError:
        return False


def _parse_choice_side(action, n_actions=N_ACTIONS):
    _validate_two_action_task(n_actions)
    try:
        if np.isnan(action):
            return None
    except TypeError:
        pass

    try:
        action = int(action)
    except (TypeError, ValueError):
        return None

    if action not in [ChoiceSide.RIGHT, ChoiceSide.LEFT]:
        return None

    return ChoiceSide(action)


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
    if action == ChoiceSide.RIGHT:
        p[ChoiceSide.RIGHT] = active_reward_probability
        p[ChoiceSide.LEFT] = inactive_reward_probability
    elif action == ChoiceSide.LEFT:
        p[ChoiceSide.RIGHT] = inactive_reward_probability
        p[ChoiceSide.LEFT] = active_reward_probability
    return p


def _nonzero_reward_size_probability(
    reward, action, correct_reward_size, incorrect_reward_size
):
    p = np.zeros(2)
    if action == ChoiceSide.RIGHT:
        p[ChoiceSide.RIGHT] = float(reward == correct_reward_size)
        p[ChoiceSide.LEFT] = float(reward == incorrect_reward_size)
    elif action == ChoiceSide.LEFT:
        p[ChoiceSide.RIGHT] = float(reward == incorrect_reward_size)
        p[ChoiceSide.LEFT] = float(reward == correct_reward_size)
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
    sign = _choices_to_signed_left_positive(choice_side)

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
    y = _choices_to_signed_left_positive(choices)

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
    y = _choices_to_signed_left_positive(choices)

    alpha = np.exp(-decay)

    pers = np.zeros(T)
    if T <= 1:
        return pers

    weights = alpha ** np.arange(1, T + 1)
    # For pers[t], only choices < t are used. Convolution index t-1 contains
    # y[t-1]*alpha + y[t-2]*alpha**2 + ... + y[0]*alpha**t, so recent choices
    # have larger weights. This mirrors the iterative recurrence above.
    history_num = np.convolve(y, weights, mode="full")[:T]
    history_den = np.convolve(np.ones(T), weights, mode="full")[:T]
    pers[1:] = history_num[:-1] / history_den[:-1]

    return pers


def _choices_to_signed_left_positive(choice_side):
    """
    Map choices to sign convention used by relative-value features:
    left -> +1, right -> -1.
    """
    choice_arr = np.asarray(choice_side)
    if choice_arr.ndim == 0:
        return _choice_token_to_signed_left_positive(choice_arr.item())

    signed = np.empty(choice_arr.shape, dtype=float)
    for idx, token in np.ndenumerate(choice_arr):
        signed[idx] = _choice_token_to_signed_left_positive(token)
    return signed


def _choice_side_to_signed_left_positive(side: ChoiceSide) -> float:
    return SIGNED_LEFT if side == ChoiceSide.LEFT else SIGNED_RIGHT


def _choice_token_to_signed_left_positive(token):
    # String labels
    if isinstance(token, str):
        parsed = token.strip().lower()
        if parsed in LEFT_CHOICE_LABELS:
            return SIGNED_LEFT
        if parsed in RIGHT_CHOICE_LABELS:
            return SIGNED_RIGHT
        try:
            token = float(parsed)
        except ValueError as exc:
            raise ValueError(
                "choice values must be left/right, 0/1, or -1/+1."
            ) from exc

    # Missing values
    try:
        if np.isnan(token):
            raise ValueError("choice values cannot be NaN.")
    except TypeError:
        pass

    # Numeric task-coded or signed choices
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("choice values must be left/right, 0/1, or -1/+1.") from exc

    if value == float(ChoiceSide.RIGHT) or value == SIGNED_RIGHT:
        return SIGNED_RIGHT
    if value == float(ChoiceSide.LEFT):
        # Numeric +1 is both task-coded left and signed left.
        return SIGNED_LEFT

    raise ValueError("choice values must be left/right, 0/1, or -1/+1.")

