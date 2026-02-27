import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import autograd.numpy as np
    import autograd.numpy.random as npr
    import matplotlib.pyplot as plt

    try:
        import ssm
        from ssm.util import find_permutation
    except ImportError as exc:
        raise ImportError(
            "The ssm package is required. Install the fork used by this tutorial."
        ) from exc

    npr.seed(0)

    def model_log_prob(model, data, inputs):
        if hasattr(model, "log_probability"):
            return model.log_probability(data, inputs=inputs)
        return model.log_likelihood(data, inputs=inputs)

    return find_permutation, model_log_prob, np, npr, plt, ssm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Input Driven Observations (GLM-HMM)

    Port of `2b Input Driven Observations (GLM-HMM).ipynb` to marimo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup
    """)
    return


@app.cell
def _():
    cols = ["#ff7f00", "#4daf4a", "#377eb8"]
    return (cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Input Driven Observations

    ### 2a. Initialize GLM-HMM
    """)
    return


@app.cell
def _(ssm):
    # Set the parameters of the GLM-HMM
    num_states = 3
    obs_dim = 1
    num_categories = 2
    input_dim = 2

    true_glmhmm = ssm.HMM(
        num_states,
        obs_dim,
        input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories),
        transitions="standard",
    )

    return input_dim, num_categories, num_states, obs_dim, true_glmhmm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2b. Specify Parameters of Generative GLM-HMM
    """)
    return


@app.cell
def _(np, true_glmhmm):
    gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
    gen_log_trans_mat = np.log(
        np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]])
    )
    true_glmhmm.observations.params = gen_weights
    true_glmhmm.transitions.params = gen_log_trans_mat

    return gen_log_trans_mat, gen_weights


@app.cell
def _(cols, gen_log_trans_mat, gen_weights, input_dim, num_states, np, plt):
    fig = plt.figure(figsize=(8, 3), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 2, 1)
    for state_idx in range(num_states):
        plt.plot(
            range(input_dim),
            gen_weights[state_idx][0],
            marker="o",
            color=cols[state_idx],
            linestyle="-",
            lw=1.5,
            label="state " + str(state_idx + 1),
        )
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], ["stimulus", "bias"], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title("Generative weights", fontsize=15)

    plt.subplot(1, 2, 2)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(gen_trans_mat.shape[0]):
        for col_idx in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(gen_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.yticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("Generative transition matrix", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c. Create External Input Sequences
    """)
    return


@app.cell
def _(input_dim, np):
    num_sess = 20
    num_trials_per_sess = 100
    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    inpts[:, :, 0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))
    inpts = list(inpts)

    return inpts, num_sess, num_trials_per_sess, stim_vals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2d. Simulate States and Observations with Generative Model
    """)
    return


@app.cell
def _(inpts, num_sess, num_trials_per_sess, true_glmhmm):
    true_latents, true_choices = [], []
    for sess_idx in range(num_sess):
        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess_idx])
        true_latents.append(true_z)
        true_choices.append(true_y)

    return true_choices, true_latents


@app.cell
def _(inpts, model_log_prob, true_choices, true_glmhmm):
    true_ll = model_log_prob(true_glmhmm, true_choices, inpts)
    print("true ll = " + str(true_ll))
    return (true_ll,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fit GLM-HMM and Perform Recovery Analysis

    ### 3a. Maximum Likelihood Estimation
    """)
    return


@app.cell
def _(input_dim, num_categories, num_states, obs_dim, ssm, true_choices, inpts):
    new_glmhmm = ssm.HMM(
        num_states,
        obs_dim,
        input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories),
        transitions="standard",
    )

    n_iters = 200
    fit_ll = new_glmhmm.fit(
        true_choices,
        inputs=inpts,
        method="em",
        num_iters=n_iters,
        tolerance=10**-4,
    )

    return fit_ll, n_iters, new_glmhmm


@app.cell
def _(fit_ll, np, plt, true_ll):
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(fit_ll, label="EM")
    plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ":k", label="True")
    plt.legend(loc="lower right")
    plt.xlabel("EM Iteration")
    plt.xlim(0, len(fit_ll))
    plt.ylabel("Log Probability")
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3b. Retrieved Parameters
    """)
    return


@app.cell
def _(find_permutation, inpts, new_glmhmm, true_choices, true_latents):
    new_glmhmm.permute(
        find_permutation(
            true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])
        )
    )
    return


@app.cell
def _(cols, gen_weights, input_dim, new_glmhmm, num_states, plt):
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    recovered_weights = new_glmhmm.observations.params
    for state_idx in range(num_states):
        if state_idx == 0:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx][0],
                marker="o",
                color=cols[state_idx],
                linestyle="-",
                lw=1.5,
                label="generative",
            )
            plt.plot(
                range(input_dim),
                recovered_weights[state_idx][0],
                color=cols[state_idx],
                lw=1.5,
                label="recovered",
                linestyle="--",
            )
        else:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx][0],
                marker="o",
                color=cols[state_idx],
                linestyle="-",
                lw=1.5,
                label="",
            )
            plt.plot(
                range(input_dim),
                recovered_weights[state_idx][0],
                color=cols[state_idx],
                lw=1.5,
                label="",
                linestyle="--",
            )
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], ["stimulus", "bias"], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title("Weight recovery", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(gen_log_trans_mat, new_glmhmm, num_states, np, plt):
    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 2, 1)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(gen_trans_mat.shape[0]):
        for col_idx in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(gen_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.yticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("generative", fontsize=15)

    plt.subplot(1, 2, 2)
    recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(recovered_trans_mat.shape[0]):
        for col_idx in range(recovered_trans_mat.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(recovered_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.yticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c. Posterior State Probabilities
    """)
    return


@app.cell
def _(inpts, new_glmhmm, true_choices):
    posterior_probs = [
        new_glmhmm.expected_states(data=data, input=inpt)[0]
        for data, inpt in zip(true_choices, inpts)
    ]
    return (posterior_probs,)


@app.cell
def _(cols, num_states, posterior_probs, plt):
    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor="w", edgecolor="k")
    sess_id = 0
    for state_idx in range(num_states):
        plt.plot(
            posterior_probs[sess_id][:, state_idx],
            label="State " + str(state_idx + 1),
            lw=2,
            color=cols[state_idx],
        )
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.xlabel("trial #", fontsize=15)
    plt.ylabel("p(state)", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(np, posterior_probs):
    posterior_probs_concat = np.concatenate(posterior_probs)
    state_max_posterior = np.argmax(posterior_probs_concat, axis=1)
    _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
    state_occupancies = state_occupancies / np.sum(state_occupancies)

    return (state_occupancies,)


@app.cell
def _(cols, plt, state_occupancies):
    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor="w", edgecolor="k")
    for state_idx, occ in enumerate(state_occupancies):
        plt.bar(state_idx, occ, width=0.8, color=cols[state_idx])
    plt.ylim((0, 1))
    plt.xticks([0, 1, 2], ["1", "2", "3"], fontsize=10)
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
    plt.xlabel("state", fontsize=15)
    plt.ylabel("frac. occupancy", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Maximum A Posteriori Estimation
    """)
    return


@app.cell
def _(input_dim, num_categories, num_states, obs_dim, ssm):
    prior_sigma = 2
    prior_alpha = 2
    map_glmhmm = ssm.HMM(
        num_states,
        obs_dim,
        input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
        transitions="sticky",
        transition_kwargs=dict(alpha=prior_alpha, kappa=0),
    )

    return map_glmhmm, prior_alpha, prior_sigma


@app.cell
def _(map_glmhmm, n_iters, true_choices, inpts):
    _ = map_glmhmm.fit(
        true_choices,
        inputs=inpts,
        method="em",
        num_iters=n_iters,
        tolerance=10**-4,
    )
    return


@app.cell
def _(inpts, map_glmhmm, new_glmhmm, true_choices, true_glmhmm):
    true_likelihood = true_glmhmm.log_likelihood(true_choices, inputs=inpts)
    mle_final_ll = new_glmhmm.log_likelihood(true_choices, inputs=inpts)
    map_final_ll = map_glmhmm.log_likelihood(true_choices, inputs=inpts)

    return map_final_ll, mle_final_ll, true_likelihood


@app.cell
def _(map_final_ll, mle_final_ll, plt, true_likelihood):
    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor="w", edgecolor="k")
    loglikelihood_vals = [true_likelihood, mle_final_ll, map_final_ll]
    colors_ll = ["Red", "Navy", "Purple"]
    for model_idx, ll_val in enumerate(loglikelihood_vals):
        plt.bar(model_idx, ll_val, width=0.8, color=colors_ll[model_idx])
    plt.ylim((true_likelihood - 5, true_likelihood + 15))
    plt.xticks([0, 1, 2], ["true", "mle", "map"], fontsize=10)
    plt.xlabel("model", fontsize=15)
    plt.ylabel("loglikelihood", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cross Validation (Held-Out Comparison)
    """)
    return


@app.cell
def _(input_dim, num_trials_per_sess, np, stim_vals):
    num_test_sess = 10
    test_inpts = np.ones((num_test_sess, num_trials_per_sess, input_dim))
    test_inpts[:, :, 0] = np.random.choice(stim_vals, (num_test_sess, num_trials_per_sess))
    test_inpts = list(test_inpts)

    return num_test_sess, test_inpts


@app.cell
def _(num_test_sess, num_trials_per_sess, test_inpts, true_glmhmm):
    test_latents, test_choices = [], []
    for sess_idx in range(num_test_sess):
        test_z, test_y = true_glmhmm.sample(num_trials_per_sess, input=test_inpts[sess_idx])
        test_latents.append(test_z)
        test_choices.append(test_y)

    return test_choices, test_latents


@app.cell
def _(map_glmhmm, new_glmhmm, test_choices, test_inpts):
    mle_test_ll = new_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
    map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpts)

    return map_test_ll, mle_test_ll


@app.cell
def _(map_test_ll, mle_test_ll, plt):
    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor="w", edgecolor="k")
    loglikelihood_vals = [mle_test_ll, map_test_ll]
    colors_ll = ["Navy", "Purple"]
    for model_idx, ll_val in enumerate(loglikelihood_vals):
        plt.bar(model_idx, ll_val, width=0.8, color=colors_ll[model_idx])
    plt.ylim((mle_test_ll - 2, mle_test_ll + 5))
    plt.xticks([0, 1], ["mle", "map"], fontsize=10)
    plt.xlabel("model", fontsize=15)
    plt.ylabel("loglikelihood", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(find_permutation, inpts, map_glmhmm, true_choices, true_latents):
    map_glmhmm.permute(
        find_permutation(
            true_latents[0], map_glmhmm.most_likely_states(true_choices[0], input=inpts[0])
        )
    )
    return


@app.cell
def _(cols, gen_weights, input_dim, map_glmhmm, new_glmhmm, num_states, plt):
    fig = plt.figure(figsize=(6, 3), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 2, 1)
    recovered_weights_mle = new_glmhmm.observations.params
    for state_idx in range(num_states):
        if state_idx == 0:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx][0],
                marker="o",
                color=cols[state_idx],
                lw=1.5,
                label="generative",
            )
            plt.plot(
                range(input_dim),
                recovered_weights_mle[state_idx][0],
                color=cols[state_idx],
                lw=1.5,
                label="recovered",
                linestyle="--",
            )
        else:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx][0],
                marker="o",
                color=cols[state_idx],
                lw=1.5,
                label="",
            )
            plt.plot(
                range(input_dim),
                recovered_weights_mle[state_idx][0],
                color=cols[state_idx],
                lw=1.5,
                label="",
                linestyle="--",
            )
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], ["stimulus", "bias"], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.title("MLE", fontsize=15)
    plt.legend()

    plt.subplot(1, 2, 2)
    recovered_weights_map = map_glmhmm.observations.params
    for state_idx in range(num_states):
        plt.plot(
            range(input_dim),
            gen_weights[state_idx][0],
            marker="o",
            color=cols[state_idx],
            lw=1.5,
            label="",
            linestyle="-",
        )
        plt.plot(
            range(input_dim),
            recovered_weights_map[state_idx][0],
            color=cols[state_idx],
            lw=1.5,
            label="",
            linestyle="--",
        )
    plt.yticks(fontsize=10)
    plt.xticks([0, 1], ["", ""], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.title("MAP", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(gen_log_trans_mat, map_glmhmm, new_glmhmm, num_states, np, plt):
    fig = plt.figure(figsize=(7, 2.5), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 3, 1)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(gen_trans_mat.shape[0]):
        for col_idx in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(gen_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.yticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("generative", fontsize=15)

    plt.subplot(1, 3, 2)
    recovered_trans_mat_mle = np.exp(new_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat_mle, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(recovered_trans_mat_mle.shape[0]):
        for col_idx in range(recovered_trans_mat_mle.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(recovered_trans_mat_mle[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.yticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered - MLE", fontsize=15)

    plt.subplot(1, 3, 3)
    recovered_trans_mat_map = np.exp(map_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat_map, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(recovered_trans_mat_map.shape[0]):
        for col_idx in range(recovered_trans_mat_map.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(recovered_trans_mat_map[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.yticks(range(num_states), ("1", "2", "3"), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered - MAP", fontsize=15)

    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Multinomial GLM-HMM
    """)
    return


@app.cell
def _(ssm):
    num_states_multi = 4
    obs_dim_multi = 1
    num_categories_multi = 3
    input_dim_multi = 2

    true_glmhmm_multi = ssm.HMM(
        num_states_multi,
        obs_dim_multi,
        input_dim_multi,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories_multi),
        transitions="standard",
    )

    return (
        input_dim_multi,
        num_categories_multi,
        num_states_multi,
        obs_dim_multi,
        true_glmhmm_multi,
    )


@app.cell
def _(np, true_glmhmm_multi):
    gen_weights_multi = np.array(
        [
            [[0.6, 3], [2, 3]],
            [[6, 1], [6, -2]],
            [[1, 1], [3, 1]],
            [[2, 2], [0, 5]],
        ]
    )
    print(gen_weights_multi.shape)
    true_glmhmm_multi.observations.params = gen_weights_multi

    return (gen_weights_multi,)


@app.cell
def _(np, true_glmhmm_multi):
    gen_log_trans_mat_multi = np.log(
        np.array(
            [
                [
                    [0.90, 0.04, 0.05, 0.01],
                    [0.05, 0.92, 0.01, 0.02],
                    [0.03, 0.02, 0.94, 0.01],
                    [0.09, 0.01, 0.01, 0.89],
                ]
            ]
        )
    )
    true_glmhmm_multi.transitions.params = gen_log_trans_mat_multi

    return (gen_log_trans_mat_multi,)


@app.cell
def _(input_dim_multi, np):
    num_sess_multi = 20
    num_trials_per_sess_multi = 1000
    inpts_multi = np.ones((num_sess_multi, num_trials_per_sess_multi, input_dim_multi))
    stim_vals_multi = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    inpts_multi[:, :, 0] = np.random.choice(
        stim_vals_multi, (num_sess_multi, num_trials_per_sess_multi)
    )
    inpts_multi = list(inpts_multi)

    return inpts_multi, num_sess_multi, num_trials_per_sess_multi


@app.cell
def _(inpts_multi, num_sess_multi, num_trials_per_sess_multi, true_glmhmm_multi):
    true_latents_multi, true_choices_multi = [], []
    for sess_idx in range(num_sess_multi):
        true_z, true_y = true_glmhmm_multi.sample(
            num_trials_per_sess_multi, input=inpts_multi[sess_idx]
        )
        true_latents_multi.append(true_z)
        true_choices_multi.append(true_y)

    return true_choices_multi, true_latents_multi


@app.cell
def _(plt, true_choices_multi):
    fig = plt.figure(figsize=(8, 3), dpi=80, facecolor="w", edgecolor="k")
    plt.step(range(100), true_choices_multi[0][range(100)], color="red")
    plt.yticks([0, 1, 2])
    plt.title("example data (multinomial GLM-HMM)")
    plt.xlabel("trial #", fontsize=15)
    plt.ylabel("observation class", fontsize=15)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(inpts_multi, model_log_prob, true_choices_multi, true_glmhmm_multi):
    true_ll_multi = model_log_prob(true_glmhmm_multi, true_choices_multi, inpts_multi)
    print("true ll = " + str(true_ll_multi))

    return (true_ll_multi,)


@app.cell
def _(
    input_dim_multi,
    num_categories_multi,
    num_states_multi,
    obs_dim_multi,
    ssm,
    true_choices_multi,
    inpts_multi,
):
    new_glmhmm_multi = ssm.HMM(
        num_states_multi,
        obs_dim_multi,
        input_dim_multi,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories_multi),
        transitions="standard",
    )

    n_iters_multi = 500
    fit_ll_multi = new_glmhmm_multi.fit(
        true_choices_multi,
        inputs=inpts_multi,
        method="em",
        num_iters=n_iters_multi,
        tolerance=10**-4,
    )

    return fit_ll_multi, new_glmhmm_multi


@app.cell
def _(fit_ll_multi, np, plt, true_ll_multi):
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(fit_ll_multi, label="EM")
    plt.plot([0, len(fit_ll_multi)], true_ll_multi * np.ones(2), ":k", label="True")
    plt.legend(loc="lower right")
    plt.xlabel("EM Iteration")
    plt.xlim(0, len(fit_ll_multi))
    plt.ylabel("Log Probability")
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(find_permutation, inpts_multi, new_glmhmm_multi, true_choices_multi, true_latents_multi):
    new_glmhmm_multi.permute(
        find_permutation(
            true_latents_multi[0],
            new_glmhmm_multi.most_likely_states(true_choices_multi[0], input=inpts_multi[0]),
        )
    )

    return


@app.cell
def _(
    gen_log_trans_mat_multi,
    gen_weights_multi,
    input_dim_multi,
    num_categories_multi,
    num_states_multi,
    np,
    plt,
    new_glmhmm_multi,
):
    recovered_weights = new_glmhmm_multi.observations.params
    recovered_transitions = new_glmhmm_multi.transitions.params

    fig = plt.figure(figsize=(16, 8), dpi=80, facecolor="w", edgecolor="k")
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    cols_multi = [
        "#ff7f00",
        "#4daf4a",
        "#377eb8",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    for class_idx in range(num_categories_multi):
        plt.subplot(2, num_categories_multi + 1, class_idx + 1)
        if class_idx < num_categories_multi - 1:
            for state_idx in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    gen_weights_multi[state_idx, class_idx],
                    marker="o",
                    color=cols_multi[state_idx],
                    lw=1.5,
                    label=f"state {state_idx + 1}; class {class_idx + 1}",
                )
        else:
            for state_idx in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    np.zeros(input_dim_multi),
                    marker="o",
                    color=cols_multi[state_idx],
                    lw=1.5,
                    label=f"state {state_idx + 1}; class {class_idx + 1}",
                    alpha=0.5,
                )

        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.yticks(fontsize=10)
        plt.xticks([0, 1], ["", ""])
        if class_idx == 0:
            plt.ylabel("GLM weight", fontsize=15)
        plt.legend()
        plt.title("Generative weights; class " + str(class_idx + 1), fontsize=15)
        plt.ylim((-3, 10))

    plt.subplot(2, num_categories_multi + 1, num_categories_multi + 1)
    gen_trans_mat = np.exp(gen_log_trans_mat_multi)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(gen_trans_mat.shape[0]):
        for col_idx in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(gen_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states_multi - 0.5)
    plt.xticks(range(num_states_multi), ("1", "2", "3", "4"), fontsize=10)
    plt.yticks(range(num_states_multi), ("1", "2", "3", "4"), fontsize=10)
    plt.ylim(num_states_multi - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("Generative transition matrix", fontsize=15)

    for class_idx in range(num_categories_multi):
        plt.subplot(2, num_categories_multi + 1, num_categories_multi + class_idx + 2)
        if class_idx < num_categories_multi - 1:
            for state_idx in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    recovered_weights[state_idx, class_idx],
                    marker="o",
                    linestyle="--",
                    color=cols_multi[state_idx],
                    lw=1.5,
                    label=f"state {state_idx + 1}; class {class_idx + 1}",
                )
        else:
            for state_idx in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    np.zeros(input_dim_multi),
                    marker="o",
                    linestyle="--",
                    color=cols_multi[state_idx],
                    lw=1.5,
                    label=f"state {state_idx + 1}; class {class_idx + 1}",
                    alpha=0.5,
                )

        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.yticks(fontsize=10)
        plt.xlabel("covariate", fontsize=15)
        if class_idx == 0:
            plt.ylabel("GLM weight", fontsize=15)
        plt.xticks([0, 1], ["stimulus", "bias"], fontsize=12, rotation=45)
        plt.legend()
        plt.title("Recovered weights; class " + str(class_idx + 1), fontsize=15)
        plt.ylim((-3, 10))

    plt.subplot(2, num_categories_multi + 1, 2 * num_categories_multi + 2)
    recovered_trans_mat = np.exp(recovered_transitions)[0]
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx in range(recovered_trans_mat.shape[0]):
        for col_idx in range(recovered_trans_mat.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(np.around(recovered_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    plt.xlim(-0.5, num_states_multi - 0.5)
    plt.xticks(range(num_states_multi), ("1", "2", "3", "4"), fontsize=10)
    plt.yticks(range(num_states_multi), ("1", "2", "3", "4"), fontsize=10)
    plt.ylim(num_states_multi - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("Recovered transition matrix", fontsize=15)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
