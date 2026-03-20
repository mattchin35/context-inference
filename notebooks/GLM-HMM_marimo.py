import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import autograd.numpy as np
    import autograd.numpy.random as npr
    import itertools
    import matplotlib.pyplot as plt
    import multiprocessing
    from pathlib import Path

    from joblib import Parallel, delayed
    from matplotlib.patches import Patch
    from sklearn.model_selection import StratifiedKFold

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

    fig_prefix = "GLM-HMM"
    fig_dir = Path(__file__).resolve().parent / "notebook_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(plt, "_notebook_orig_show"):
        plt._notebook_orig_show = plt.show
    if not hasattr(plt, "_notebook_save_enabled"):
        plt._notebook_save_enabled = True
    if not hasattr(plt, "_notebook_fig_counter_by_prefix"):
        plt._notebook_fig_counter_by_prefix = {}

    def configure_figure_saving(enabled):
        plt._notebook_save_enabled = bool(enabled)

    def _save_open_figures_with_prefix():
        if fig_prefix not in plt._notebook_fig_counter_by_prefix:
            plt._notebook_fig_counter_by_prefix[fig_prefix] = itertools.count(1)

        counter_prefix = plt._notebook_fig_counter_by_prefix[fig_prefix]
        for fig_num_prefix in plt.get_fignums():
            fig_obj_prefix = plt.figure(fig_num_prefix)
            fig_idx_prefix = next(counter_prefix)
            out_path_prefix = fig_dir / f"{fig_prefix}_{fig_idx_prefix:03d}.png"
            fig_obj_prefix.savefig(out_path_prefix, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {out_path_prefix}")

    def _show_and_save_prefix(*args, **kwargs):
        if getattr(plt, "_notebook_save_enabled", True):
            _save_open_figures_with_prefix()
        return plt._notebook_orig_show(*args, **kwargs)

    plt.show = _show_and_save_prefix
    configure_figure_saving(True)
    return (
        Parallel,
        Patch,
        StratifiedKFold,
        configure_figure_saving,
        delayed,
        find_permutation,
        model_log_prob,
        multiprocessing,
        np,
        plt,
        ssm,
    )


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
    save_figures = True
    return (save_figures,)


@app.cell(hide_code=True)
def _(mo, save_figures):
    mo.md(rf"""
    **Figure Saving**

    - `save_figures = {save_figures}`
    - output directory: `notebooks/notebook_figures`
    """)
    return


@app.cell
def _(configure_figure_saving, save_figures):
    configure_figure_saving(save_figures)
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
def _(cols, gen_log_trans_mat, gen_weights, input_dim, np, num_states, plt):
    fig = plt.figure(figsize=(8, 3), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 2, 1)
    for state_idx_s2plot in range(num_states):
        plt.plot(
            range(input_dim),
            gen_weights[state_idx_s2plot][0],
            marker="o",
            color=cols[state_idx_s2plot],
            linestyle="-",
            lw=1.5,
            label="state " + str(state_idx_s2plot + 1),
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
    for row_idx_s2plot in range(gen_trans_mat.shape[0]):
        for col_idx_s2plot in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx_s2plot,
                row_idx_s2plot,
                str(np.around(gen_trans_mat[row_idx_s2plot, col_idx_s2plot], decimals=2)),
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
    return (gen_trans_mat,)


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
    for sess_idx_synth in range(num_sess):
        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess_idx_synth])
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
def _(
    inpts,
    input_dim,
    num_categories,
    num_states,
    obs_dim,
    ssm,
    true_choices,
):
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
    _ = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
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
    _ = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    recovered_weights = new_glmhmm.observations.params
    for state_idx_recovery in range(num_states):
        if state_idx_recovery == 0:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx_recovery][0],
                marker="o",
                color=cols[state_idx_recovery],
                linestyle="-",
                lw=1.5,
                label="generative",
            )
            plt.plot(
                range(input_dim),
                recovered_weights[state_idx_recovery][0],
                color=cols[state_idx_recovery],
                lw=1.5,
                label="recovered",
                linestyle="--",
            )
        else:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx_recovery][0],
                marker="o",
                color=cols[state_idx_recovery],
                linestyle="-",
                lw=1.5,
                label="",
            )
            plt.plot(
                range(input_dim),
                recovered_weights[state_idx_recovery][0],
                color=cols[state_idx_recovery],
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
def _(gen_log_trans_mat, gen_trans_mat, new_glmhmm, np, num_states, plt):
    _ = plt.figure(figsize=(5, 2.5), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 2, 1)
    gen_trans_mat_ = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat_, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx_transcmp in range(gen_trans_mat_.shape[0]):
        for col_idx_transcmp in range(gen_trans_mat_.shape[1]):
            plt.text(
                col_idx_transcmp,
                row_idx_transcmp,
                str(np.around(gen_trans_mat[row_idx_transcmp, col_idx_transcmp], decimals=2)),
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
    recovered_trans_mat_ = np.exp(new_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat_, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx_transcmp in range(recovered_trans_mat_.shape[0]):
        for col_idx_transcmp in range(recovered_trans_mat_.shape[1]):
            plt.text(
                col_idx_transcmp,
                row_idx_transcmp,
                str(np.around(recovered_trans_mat_[row_idx_transcmp, col_idx_transcmp], decimals=2)),
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
def _(cols, num_states, plt, posterior_probs):
    _ = plt.figure(figsize=(5, 2.5), dpi=80, facecolor="w", edgecolor="k")
    sess_id = 0
    for state_idx_post in range(num_states):
        plt.plot(
            posterior_probs[sess_id][:, state_idx_post],
            label="State " + str(state_idx_post + 1),
            lw=2,
            color=cols[state_idx_post],
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
    _ = plt.figure(figsize=(2, 2.5), dpi=80, facecolor="w", edgecolor="k")
    for state_idx_occ, occ_occ in enumerate(state_occupancies):
        plt.bar(state_idx_occ, occ_occ, width=0.8, color=cols[state_idx_occ])
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
    return (map_glmhmm,)


@app.cell
def _(inpts, map_glmhmm, n_iters, true_choices):
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
    _ = plt.figure(figsize=(2, 2.5), dpi=80, facecolor="w", edgecolor="k")
    loglikelihood_vals = [true_likelihood, mle_final_ll, map_final_ll]
    colors_ll = ["Red", "Navy", "Purple"]
    for model_idx_fitcmp, ll_val_fitcmp in enumerate(loglikelihood_vals):
        plt.bar(model_idx_fitcmp, ll_val_fitcmp, width=0.8, color=colors_ll[model_idx_fitcmp])
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
    ## 5. StratifiedKFold Cross-Validation Model Selection

    Compare MLE and MAP held-out log-likelihood over candidate hidden-state
    counts using session-stratified folds.
    """)
    return


@app.cell
def _(multiprocessing):
    # Cross-validation runtime controls.
    run_model_selection = True
    max_states_cv = 4
    n_iters_cv = 1000
    tol_cv = 1e-4
    n_run_em_cv = 4
    n_kfold_cv = 4
    num_threads_cv = max(1, (multiprocessing.cpu_count() - 1) * 2)

    # MAP prior settings used inside CV.
    prior_alpha_cv = 2
    prior_sigma_cv = 2

    # Information-criterion runtime controls.
    run_information_criteria = True
    n_run_em_ic = 4
    n_iters_ic = 1000
    tol_ic = 1e-4
    return (
        max_states_cv,
        n_iters_cv,
        n_iters_ic,
        n_kfold_cv,
        n_run_em_cv,
        n_run_em_ic,
        num_threads_cv,
        prior_alpha_cv,
        prior_sigma_cv,
        run_information_criteria,
        run_model_selection,
        tol_cv,
        tol_ic,
    )


@app.cell(hide_code=True)
def _(
    max_states_cv,
    mo,
    n_iters_cv,
    n_iters_ic,
    n_kfold_cv,
    n_run_em_cv,
    n_run_em_ic,
    num_threads_cv,
    prior_alpha_cv,
    prior_sigma_cv,
    run_information_criteria,
    run_model_selection,
    tol_cv,
    tol_ic,
):
    mo.md(rf"""
    **Section 5 Controls**

    - `run_model_selection = {run_model_selection}`
    - `run_information_criteria = {run_information_criteria}`
    - `max_states_cv = {max_states_cv}`
    - `n_kfold_cv = {n_kfold_cv}`
    - `n_run_em_cv = {n_run_em_cv}`
    - `n_run_em_ic = {n_run_em_ic}`
    - `n_iters_cv = {n_iters_cv}`
    - `n_iters_ic = {n_iters_ic}`
    - `tol_cv = {tol_cv}`
    - `tol_ic = {tol_ic}`
    - `num_threads_cv = {num_threads_cv}`
    - `prior_alpha_cv = {prior_alpha_cv}`
    - `prior_sigma_cv = {prior_sigma_cv}`

    Edit the control cell above to change model-selection behavior.
    """)
    return


@app.cell
def _(inpts, np, num_sess, true_choices):
    synthetic_data_cv = np.vstack(true_choices)
    synthetic_inpts_cv = np.vstack(inpts)
    trials_per_sess_cv = len(true_choices[0])
    session_labels_cv = np.repeat(np.arange(num_sess), trials_per_sess_cv)
    return session_labels_cv, synthetic_data_cv, synthetic_inpts_cv


@app.cell
def _(
    Patch,
    StratifiedKFold,
    n_kfold_cv,
    np,
    plt,
    session_labels_cv,
    synthetic_data_cv,
):
    min_class_count_cv = int(np.min(np.bincount(session_labels_cv)))
    nKfold_cv = max(2, min(n_kfold_cv, min_class_count_cv))

    cv_for_plot_cv = StratifiedKFold(n_splits=nKfold_cv, shuffle=True, random_state=0)
    fig_cvplot, ax_cvplot = plt.subplots(figsize=(6, 3))
    cmap_cvplot = plt.cm.coolwarm
    cmap_data_cvplot = plt.cm.Paired

    for split_idx_cvplot, (train_idx_cvplot, test_idx_cvplot) in enumerate(
        cv_for_plot_cv.split(X=synthetic_data_cv, y=session_labels_cv)
    ):
        index_marks_cvplot = np.array([np.nan] * len(synthetic_data_cv))
        index_marks_cvplot[test_idx_cvplot] = 1
        index_marks_cvplot[train_idx_cvplot] = 0
        ax_cvplot.scatter(
            range(len(index_marks_cvplot)),
            [split_idx_cvplot + 0.5] * len(index_marks_cvplot),
            c=index_marks_cvplot,
            marker="_",
            lw=8,
            cmap=cmap_cvplot,
            vmin=-0.2,
            vmax=1.2,
        )

    ax_cvplot.scatter(
        range(len(synthetic_data_cv)),
        [nKfold_cv + 0.5] * len(synthetic_data_cv),
        c=session_labels_cv,
        marker="_",
        lw=8,
        cmap=cmap_data_cvplot,
    )
    ax_cvplot.set(
        yticks=np.arange(nKfold_cv + 1) + 0.5,
        yticklabels=list(range(nKfold_cv)) + ["session"],
        xlabel="trial index",
        ylabel="CV iteration",
        ylim=[nKfold_cv + 1.2, -0.2],
    )
    ax_cvplot.set_title("StratifiedKFold", fontsize=14)
    ax_cvplot.legend(
        [Patch(color=cmap_cvplot(0.8)), Patch(color=cmap_cvplot(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    fig_cvplot.tight_layout()
    fig_cvplot.subplots_adjust(right=0.75)
    plt.show()
    return (nKfold_cv,)


@app.cell
def _(model_log_prob, ssm):
    def build_input_driven_glmhmm_cv(
        num_states_cvlocal,
        obs_dim_cvlocal,
        input_dim_cvlocal,
        num_categories_cvlocal,
        algorithm_cvlocal="MLE",
        prior_alpha_cvlocal=2,
        prior_sigma_cvlocal=2,
    ):
        algorithm_cvlocal = algorithm_cvlocal.upper()
        if algorithm_cvlocal == "MLE":
            return ssm.HMM(
                num_states_cvlocal,
                obs_dim_cvlocal,
                input_dim_cvlocal,
                observations="input_driven_obs",
                observation_kwargs=dict(C=num_categories_cvlocal),
                transitions="standard",
            )
        if algorithm_cvlocal == "MAP":
            return ssm.HMM(
                num_states_cvlocal,
                obs_dim_cvlocal,
                input_dim_cvlocal,
                observations="input_driven_obs",
                observation_kwargs=dict(C=num_categories_cvlocal, prior_sigma=prior_sigma_cvlocal),
                transitions="sticky",
                transition_kwargs=dict(alpha=prior_alpha_cvlocal, kappa=0),
            )
        raise ValueError(f"Unknown algorithm: {algorithm_cvlocal}")

    def xval_func_glmcv(data_in_glmcv, num_states_glmcv):
        training_data_glmcv = data_in_glmcv["training_data"]
        test_data_glmcv = data_in_glmcv["test_data"]
        training_inpts_glmcv = data_in_glmcv["training_inpts"]
        test_inpts_glmcv = data_in_glmcv["test_inpts"]
        n_iters_glmcv = data_in_glmcv["N_iters"]
        tol_glmcv = data_in_glmcv["TOL"]
        num_categories_glmcv = data_in_glmcv["num_categories"]
        prior_alpha_glmcv = data_in_glmcv["prior_alpha"]
        prior_sigma_glmcv = data_in_glmcv["prior_sigma"]

        obs_dim_glmcv = len(training_data_glmcv[0])
        input_dim_glmcv = len(training_inpts_glmcv[0])
        n_train_glmcv = len(training_data_glmcv)
        n_test_glmcv = len(test_data_glmcv)

        out_glmcv = {}

        mle_hmm_glmcv = build_input_driven_glmhmm_cv(
            num_states_glmcv,
            obs_dim_glmcv,
            input_dim_glmcv,
            num_categories_glmcv,
            algorithm_cvlocal="MLE",
        )
        mle_hmm_glmcv.fit(
            training_data_glmcv,
            inputs=training_inpts_glmcv,
            method="em",
            num_iters=n_iters_glmcv,
            tolerance=tol_glmcv,
        )
        out_glmcv["ll_training"] = (
            model_log_prob(mle_hmm_glmcv, training_data_glmcv, training_inpts_glmcv) / n_train_glmcv
        )
        out_glmcv["ll_heldout"] = (
            model_log_prob(mle_hmm_glmcv, test_data_glmcv, test_inpts_glmcv) / n_test_glmcv
        )

        map_hmm_glmcv = build_input_driven_glmhmm_cv(
            num_states_glmcv,
            obs_dim_glmcv,
            input_dim_glmcv,
            num_categories_glmcv,
            algorithm_cvlocal="MAP",
            prior_alpha_cvlocal=prior_alpha_glmcv,
            prior_sigma_cvlocal=prior_sigma_glmcv,
        )
        map_hmm_glmcv.fit(
            training_data_glmcv,
            inputs=training_inpts_glmcv,
            method="em",
            num_iters=n_iters_glmcv,
            tolerance=tol_glmcv,
        )
        out_glmcv["ll_training_map"] = (
            model_log_prob(map_hmm_glmcv, training_data_glmcv, training_inpts_glmcv) / n_train_glmcv
        )
        out_glmcv["ll_heldout_map"] = (
            model_log_prob(map_hmm_glmcv, test_data_glmcv, test_inpts_glmcv) / n_test_glmcv
        )

        return out_glmcv

    def count_glmhmm_params_ic(num_states_iclocal, input_dim_iclocal, num_categories_iclocal):
        # Params: transition rows + initial-state probs + GLM weights.
        n_transition_params_iclocal = num_states_iclocal * (num_states_iclocal - 1)
        n_initial_params_iclocal = num_states_iclocal - 1
        n_obs_params_iclocal = (
            num_states_iclocal * (num_categories_iclocal - 1) * input_dim_iclocal
        )
        return n_transition_params_iclocal + n_initial_params_iclocal + n_obs_params_iclocal

    def single_ic_func_glmcv(
        observations_iclocal,
        inputs_iclocal,
        num_states_iclocal,
        num_categories_iclocal,
        n_iters_iclocal,
        tol_iclocal,
        algorithm_iclocal,
        prior_alpha_iclocal,
        prior_sigma_iclocal,
    ):
        obs_dim_iclocal = len(observations_iclocal[0])
        input_dim_iclocal = len(inputs_iclocal[0])
        hmm_iclocal = build_input_driven_glmhmm_cv(
            num_states_iclocal,
            obs_dim_iclocal,
            input_dim_iclocal,
            num_categories_iclocal,
            algorithm_cvlocal=algorithm_iclocal,
            prior_alpha_cvlocal=prior_alpha_iclocal,
            prior_sigma_cvlocal=prior_sigma_iclocal,
        )
        hmm_iclocal.fit(
            observations_iclocal,
            inputs=inputs_iclocal,
            method="em",
            num_iters=n_iters_iclocal,
            tolerance=tol_iclocal,
        )
        return model_log_prob(hmm_iclocal, observations_iclocal, inputs_iclocal)

    return count_glmhmm_params_ic, single_ic_func_glmcv, xval_func_glmcv


@app.cell
def _(
    Parallel,
    StratifiedKFold,
    delayed,
    max_states_cv,
    nKfold_cv,
    n_iters_cv,
    n_run_em_cv,
    np,
    num_categories,
    num_threads_cv,
    prior_alpha_cv,
    prior_sigma_cv,
    run_model_selection,
    session_labels_cv,
    synthetic_data_cv,
    synthetic_inpts_cv,
    tol_cv,
    xval_func_glmcv,
):
    if run_model_selection:
        ll_training_cv = np.zeros((max_states_cv, nKfold_cv, n_run_em_cv))
        ll_heldout_cv = np.zeros((max_states_cv, nKfold_cv, n_run_em_cv))
        ll_training_map_cv = np.zeros((max_states_cv, nKfold_cv, n_run_em_cv))
        ll_heldout_map_cv = np.zeros((max_states_cv, nKfold_cv, n_run_em_cv))

        state_grid_cv = np.flip(np.tile(np.arange(1, max_states_cv + 1), n_run_em_cv))
        run_grid_cv = np.repeat(np.arange(1, n_run_em_cv + 1), max_states_cv, axis=0)

        print(f"Running stratified CV with {num_threads_cv} workers")

        cv_for_grid_cv = StratifiedKFold(n_splits=nKfold_cv, shuffle=True, random_state=0)
        for fold_idx_cvgrid, (train_index_cvgrid, test_index_cvgrid) in enumerate(
            cv_for_grid_cv.split(synthetic_data_cv, session_labels_cv)
        ):
            data_in_cvgrid = {
                "training_data": synthetic_data_cv[train_index_cvgrid],
                "test_data": synthetic_data_cv[test_index_cvgrid],
                "training_inpts": synthetic_inpts_cv[train_index_cvgrid],
                "test_inpts": synthetic_inpts_cv[test_index_cvgrid],
                "num_categories": num_categories,
                "N_iters": n_iters_cv,
                "TOL": tol_cv,
                "prior_alpha": prior_alpha_cv,
                "prior_sigma": prior_sigma_cv,
            }

            results_cvgrid = Parallel(n_jobs=num_threads_cv)(
                delayed(xval_func_glmcv)(data_in_cvgrid, num_states_cvgrid)
                for run_idx_cvgrid, num_states_cvgrid in zip(run_grid_cv, state_grid_cv)
            )

            for result_idx_cvgrid in range(max_states_cv * n_run_em_cv):
                ll_training_cv[
                    state_grid_cv[result_idx_cvgrid] - 1, fold_idx_cvgrid, run_grid_cv[result_idx_cvgrid] - 1
                ] = results_cvgrid[result_idx_cvgrid]["ll_training"]
                ll_heldout_cv[
                    state_grid_cv[result_idx_cvgrid] - 1, fold_idx_cvgrid, run_grid_cv[result_idx_cvgrid] - 1
                ] = results_cvgrid[result_idx_cvgrid]["ll_heldout"]
                ll_training_map_cv[
                    state_grid_cv[result_idx_cvgrid] - 1, fold_idx_cvgrid, run_grid_cv[result_idx_cvgrid] - 1
                ] = results_cvgrid[result_idx_cvgrid]["ll_training_map"]
                ll_heldout_map_cv[
                    state_grid_cv[result_idx_cvgrid] - 1, fold_idx_cvgrid, run_grid_cv[result_idx_cvgrid] - 1
                ] = results_cvgrid[result_idx_cvgrid]["ll_heldout_map"]
    else:
        ll_training_cv = None
        ll_heldout_cv = None
        ll_training_map_cv = None
        ll_heldout_map_cv = None
        print("Model selection skipped. Set run_model_selection = True to enable.")
    return ll_heldout_cv, ll_heldout_map_cv, ll_training_cv, ll_training_map_cv


@app.cell
def _(
    ll_heldout_cv,
    ll_heldout_map_cv,
    ll_training_cv,
    ll_training_map_cv,
    max_states_cv,
    nKfold_cv,
    n_run_em_cv,
    np,
    plt,
    run_model_selection,
):
    if run_model_selection and all(
        metric_arr_cv is not None
        for metric_arr_cv in [ll_training_cv, ll_heldout_cv, ll_training_map_cv, ll_heldout_map_cv]
    ):
        fig_cvsummary = plt.figure(figsize=(10, 4), dpi=80, facecolor="w", edgecolor="k")

        ll_training_plot_cv = ll_training_cv.reshape(max_states_cv, nKfold_cv * n_run_em_cv)
        ll_heldout_plot_cv = ll_heldout_cv.reshape(max_states_cv, nKfold_cv * n_run_em_cv)
        ll_training_map_plot_cv = ll_training_map_cv.reshape(max_states_cv, nKfold_cv * n_run_em_cv)
        ll_heldout_map_plot_cv = ll_heldout_map_cv.reshape(max_states_cv, nKfold_cv * n_run_em_cv)

        cv_colors_summary = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        for state_idx_cvsummary in range(max_states_cv):
            plt.plot(
                (state_idx_cvsummary + 1) * np.ones(nKfold_cv * n_run_em_cv),
                ll_training_plot_cv[state_idx_cvsummary, :],
                color=cv_colors_summary[0],
                marker="o",
                lw=0,
                alpha=0.6,
            )
            plt.plot(
                (state_idx_cvsummary + 1) * np.ones(nKfold_cv * n_run_em_cv),
                ll_heldout_plot_cv[state_idx_cvsummary, :],
                color=cv_colors_summary[1],
                marker="o",
                lw=0,
                alpha=0.6,
            )
            plt.plot(
                (state_idx_cvsummary + 1) * np.ones(nKfold_cv * n_run_em_cv),
                ll_training_map_plot_cv[state_idx_cvsummary, :],
                color=cv_colors_summary[2],
                marker="o",
                lw=0,
                alpha=0.6,
            )
            plt.plot(
                (state_idx_cvsummary + 1) * np.ones(nKfold_cv * n_run_em_cv),
                ll_heldout_map_plot_cv[state_idx_cvsummary, :],
                color=cv_colors_summary[3],
                marker="o",
                lw=0,
                alpha=0.6,
            )

        x_states_cvsummary = range(1, max_states_cv + 1)

        y_training_mle_cvsummary = ll_training_plot_cv.mean(axis=1)
        err_training_mle_cvsummary = ll_training_plot_cv.std(axis=1)
        plt.plot(x_states_cvsummary, y_training_mle_cvsummary, label="training_MLE", color=cv_colors_summary[0])
        plt.fill_between(
            x_states_cvsummary,
            y_training_mle_cvsummary - err_training_mle_cvsummary,
            y_training_mle_cvsummary + err_training_mle_cvsummary,
            alpha=0.1,
            color=cv_colors_summary[0],
        )

        y_test_mle_cvsummary = ll_heldout_plot_cv.mean(axis=1)
        err_test_mle_cvsummary = ll_heldout_plot_cv.std(axis=1)
        plt.plot(x_states_cvsummary, y_test_mle_cvsummary, label="test_MLE", color=cv_colors_summary[1])
        plt.fill_between(
            x_states_cvsummary,
            y_test_mle_cvsummary - err_test_mle_cvsummary,
            y_test_mle_cvsummary + err_test_mle_cvsummary,
            alpha=0.1,
            color=cv_colors_summary[1],
        )

        y_training_map_cvsummary = ll_training_map_plot_cv.mean(axis=1)
        err_training_map_cvsummary = ll_training_map_plot_cv.std(axis=1)
        plt.plot(x_states_cvsummary, y_training_map_cvsummary, label="training_MAP", color=cv_colors_summary[2])
        plt.fill_between(
            x_states_cvsummary,
            y_training_map_cvsummary - err_training_map_cvsummary,
            y_training_map_cvsummary + err_training_map_cvsummary,
            alpha=0.1,
            color=cv_colors_summary[2],
        )

        y_test_map_cvsummary = ll_heldout_map_plot_cv.mean(axis=1)
        err_test_map_cvsummary = ll_heldout_map_plot_cv.std(axis=1)
        plt.plot(x_states_cvsummary, y_test_map_cvsummary, label="test_MAP", color=cv_colors_summary[3])
        plt.fill_between(
            x_states_cvsummary,
            y_test_map_cvsummary - err_test_map_cvsummary,
            y_test_map_cvsummary + err_test_map_cvsummary,
            alpha=0.1,
            color=cv_colors_summary[3],
        )

        plt.legend(loc="lower right")
        plt.xlabel("states")
        plt.xlim(0, max_states_cv + 1)
        plt.ylabel("Log-Likelihood per trial")
        plt.title("StratifiedKFold model selection")
        plt.tight_layout()
        plt.show()

        best_state_mle_cv = int(np.argmax(y_test_mle_cvsummary) + 1)
        best_state_map_cv = int(np.argmax(y_test_map_cvsummary) + 1)
        print(f"Best MLE states by held-out LL: {best_state_mle_cv}")
        print(f"Best MAP states by held-out LL: {best_state_map_cv}")
    else:
        best_state_mle_cv = None
        best_state_map_cv = None
    return best_state_map_cv, best_state_mle_cv


@app.cell(hide_code=True)
def _(best_state_map_cv, best_state_mle_cv, mo):
    if best_state_mle_cv is not None and best_state_map_cv is not None:
        mo.md(rf"""
        **CV Selection Summary**

        - Best MLE model: `{best_state_mle_cv}` state(s)
        - Best MAP model: `{best_state_map_cv}` state(s)
        """)
    else:
        mo.md(r"""
        **CV Selection Summary**

        Cross-validation is currently skipped. Enable it by setting
        `run_model_selection = True` in the control cell.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b. Information Criteria (AIC/BIC)

    Compute AIC/BIC across hidden-state counts and random restarts for MLE
    models only. Information criteria should only be run on MLE models.
    """)
    return


@app.cell
def _(
    Parallel,
    count_glmhmm_params_ic,
    delayed,
    max_states_cv,
    n_iters_ic,
    n_run_em_ic,
    np,
    num_categories,
    num_threads_cv,
    prior_alpha_cv,
    prior_sigma_cv,
    run_information_criteria,
    single_ic_func_glmcv,
    synthetic_data_cv,
    synthetic_inpts_cv,
    tol_ic,
):
    if run_information_criteria:
        state_values_ic = np.arange(1, max_states_cv + 1)
        AIC_mle_cv = np.zeros((max_states_cv, n_run_em_ic))
        BIC_mle_cv = np.zeros((max_states_cv, n_run_em_ic))

        n_timesteps_ic = len(synthetic_data_cv)
        input_dim_ic = len(synthetic_inpts_cv[0])

        for state_idx_icgrid, num_states_icgrid in enumerate(state_values_ic):
            n_params_ic = count_glmhmm_params_ic(num_states_icgrid, input_dim_ic, num_categories)

            results_mle_ic = Parallel(n_jobs=num_threads_cv)(
                delayed(single_ic_func_glmcv)(
                    synthetic_data_cv,
                    synthetic_inpts_cv,
                    num_states_icgrid,
                    num_categories,
                    n_iters_ic,
                    tol_ic,
                    "MLE",
                    prior_alpha_cv,
                    prior_sigma_cv,
                )
                for run_idx_icgrid in range(n_run_em_ic)
            )

            for arr_idx_icpack in range(n_run_em_ic):
                BIC_mle_cv[state_idx_icgrid, arr_idx_icpack] = (
                    n_params_ic * np.log(n_timesteps_ic) - 2 * results_mle_ic[arr_idx_icpack]
                )
                AIC_mle_cv[state_idx_icgrid, arr_idx_icpack] = (
                    2 * n_params_ic - 2 * results_mle_ic[arr_idx_icpack]
                )
    else:
        state_values_ic = None
        AIC_mle_cv = None
        BIC_mle_cv = None
        print("Information criteria skipped. Set run_information_criteria = True to enable.")
    return AIC_mle_cv, BIC_mle_cv, state_values_ic


@app.cell
def _(
    AIC_mle_cv,
    BIC_mle_cv,
    np,
    plt,
    run_information_criteria,
    state_values_ic,
):
    if run_information_criteria and all(
        metric_icplot is not None
        for metric_icplot in [AIC_mle_cv, BIC_mle_cv, state_values_ic]
    ):
        fig_icplot = plt.figure(figsize=(5, 4), dpi=80, facecolor="w", edgecolor="k")

        y_bic_mle_icplot = np.mean(BIC_mle_cv, axis=1)
        err_bic_mle_icplot = np.std(BIC_mle_cv, axis=1)
        y_aic_mle_icplot = np.mean(AIC_mle_cv, axis=1)
        err_aic_mle_icplot = np.std(AIC_mle_cv, axis=1)
        plt.plot(state_values_ic, y_bic_mle_icplot, label="BIC (MLE)", color="tab:orange")
        plt.fill_between(
            state_values_ic,
            y_bic_mle_icplot - err_bic_mle_icplot,
            y_bic_mle_icplot + err_bic_mle_icplot,
            alpha=0.2,
            color="tab:orange",
        )
        plt.plot(state_values_ic, y_aic_mle_icplot, label="AIC (MLE)", color="tab:blue")
        plt.fill_between(
            state_values_ic,
            y_aic_mle_icplot - err_aic_mle_icplot,
            y_aic_mle_icplot + err_aic_mle_icplot,
            alpha=0.2,
            color="tab:blue",
        )
        plt.xlabel("states")
        plt.ylabel("criterion")
        plt.title("MLE information criteria")
        plt.xticks(state_values_ic)
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()

        best_state_aic_mle_ic = int(state_values_ic[np.argmin(y_aic_mle_icplot)])
        best_state_bic_mle_ic = int(state_values_ic[np.argmin(y_bic_mle_icplot)])
        print(f"Best AIC MLE states: {best_state_aic_mle_ic}")
        print(f"Best BIC MLE states: {best_state_bic_mle_ic}")
    else:
        best_state_aic_mle_ic = None
        best_state_bic_mle_ic = None
    return best_state_aic_mle_ic, best_state_bic_mle_ic


@app.cell(hide_code=True)
def _(
    best_state_aic_mle_ic,
    best_state_bic_mle_ic,
    mo,
):
    if all(
        val_icsummary is not None
        for val_icsummary in [best_state_aic_mle_ic, best_state_bic_mle_ic]
    ):
        mo.md(rf"""
        **Information-Criterion Summary**

        Information criteria should only be run on MLE models.

        - Best AIC MLE model: `{best_state_aic_mle_ic}` state(s)
        - Best BIC MLE model: `{best_state_bic_mle_ic}` state(s)
        """)
    else:
        mo.md(r"""
        **Information-Criterion Summary**

        Information criteria should only be run on MLE models.

        Information-criterion analysis is currently skipped. Enable it by setting
        `run_information_criteria = True` in the control cell.
        """)
    return


@app.cell
def _(
    AIC_mle_cv,
    BIC_mle_cv,
    best_state_aic_mle_ic,
    best_state_bic_mle_ic,
    best_state_map_cv,
    best_state_mle_cv,
    ll_heldout_cv,
    ll_heldout_map_cv,
    ll_training_cv,
    ll_training_map_cv,
):
    model_sel_glmhmm = {
        "ll_training_cv": ll_training_cv,
        "ll_heldout_cv": ll_heldout_cv,
        "ll_training_map_cv": ll_training_map_cv,
        "ll_heldout_map_cv": ll_heldout_map_cv,
        "AIC_mle_cv": AIC_mle_cv,
        "BIC_mle_cv": BIC_mle_cv,
        "best_state_mle_cv": best_state_mle_cv,
        "best_state_map_cv": best_state_map_cv,
        "best_state_aic_mle_ic": best_state_aic_mle_ic,
        "best_state_bic_mle_ic": best_state_bic_mle_ic,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5a. Simple Held-Out MLE vs MAP Estimator Comparison

    This section is a simple held-out estimator comparison between MLE and MAP.
    It uses one held-out dataset generated from the same synthetic GLM-HMM setup.
    It is not state-count model selection.
    """)
    return


@app.cell
def _(input_dim, np, num_trials_per_sess, stim_vals):
    num_test_sess = 10
    test_inpts = np.ones((num_test_sess, num_trials_per_sess, input_dim))
    test_inpts[:, :, 0] = np.random.choice(stim_vals, (num_test_sess, num_trials_per_sess))
    test_inpts = list(test_inpts)
    return num_test_sess, test_inpts


@app.cell
def _(num_test_sess, num_trials_per_sess, test_inpts, true_glmhmm):
    test_latents, test_choices = [], []
    for _sess_idx in range(num_test_sess):
        test_z, test_y = true_glmhmm.sample(num_trials_per_sess, input=test_inpts[_sess_idx])
        test_latents.append(test_z)
        test_choices.append(test_y)
    return (test_choices,)


@app.cell
def _(map_glmhmm, new_glmhmm, test_choices, test_inpts):
    mle_test_ll = new_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
    map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpts)
    return map_test_ll, mle_test_ll


@app.cell
def _(map_test_ll, mle_test_ll, plt):
    _ = plt.figure(figsize=(2, 2.5), dpi=80, facecolor="w", edgecolor="k")
    loglikelihood_vals = [mle_test_ll, map_test_ll]
    colors_ll = ["Navy", "Purple"]
    for model_idx_holdout, ll_val_holdout in enumerate(loglikelihood_vals):
        plt.bar(model_idx_holdout, ll_val_holdout, width=0.8, color=colors_ll[model_idx_holdout])
    plt.ylim((mle_test_ll - 2, mle_test_ll + 5))
    plt.xticks([0, 1], ["MLE", "MAP"], fontsize=10)
    plt.xlabel("estimator", fontsize=15)
    plt.ylabel("held-out log likelihood", fontsize=15)
    plt.title("Held-out estimator comparison", fontsize=12)
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
    for state_idx_paramcmp in range(num_states):
        if state_idx_paramcmp == 0:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx_paramcmp][0],
                marker="o",
                color=cols[state_idx_paramcmp],
                lw=1.5,
                label="generative",
            )
            plt.plot(
                range(input_dim),
                recovered_weights_mle[state_idx_paramcmp][0],
                color=cols[state_idx_paramcmp],
                lw=1.5,
                label="recovered",
                linestyle="--",
            )
        else:
            plt.plot(
                range(input_dim),
                gen_weights[state_idx_paramcmp][0],
                marker="o",
                color=cols[state_idx_paramcmp],
                lw=1.5,
                label="",
            )
            plt.plot(
                range(input_dim),
                recovered_weights_mle[state_idx_paramcmp][0],
                color=cols[state_idx_paramcmp],
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
    for state_idx_paramcmp in range(num_states):
        plt.plot(
            range(input_dim),
            gen_weights[state_idx_paramcmp][0],
            marker="o",
            color=cols[state_idx_paramcmp],
            lw=1.5,
            label="",
            linestyle="-",
        )
        plt.plot(
            range(input_dim),
            recovered_weights_map[state_idx_paramcmp][0],
            color=cols[state_idx_paramcmp],
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
def _(gen_log_trans_mat, map_glmhmm, new_glmhmm, np, num_states, plt):
    fig = plt.figure(figsize=(7, 2.5), dpi=80, facecolor="w", edgecolor="k")

    plt.subplot(1, 3, 1)
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx_mapcmp in range(gen_trans_mat.shape[0]):
        for col_idx_mapcmp in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx_mapcmp,
                row_idx_mapcmp,
                str(np.around(gen_trans_mat[row_idx_mapcmp, col_idx_mapcmp], decimals=2)),
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
    for row_idx_mapcmp in range(recovered_trans_mat_mle.shape[0]):
        for col_idx_mapcmp in range(recovered_trans_mat_mle.shape[1]):
            plt.text(
                col_idx_mapcmp,
                row_idx_mapcmp,
                str(np.around(recovered_trans_mat_mle[row_idx_mapcmp, col_idx_mapcmp], decimals=2)),
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
    for row_idx_mapcmp in range(recovered_trans_mat_map.shape[0]):
        for col_idx_mapcmp in range(recovered_trans_mat_map.shape[1]):
            plt.text(
                col_idx_mapcmp,
                row_idx_mapcmp,
                str(np.around(recovered_trans_mat_map[row_idx_mapcmp, col_idx_mapcmp], decimals=2)),
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
    return (gen_trans_mat,)


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
def _(
    inpts_multi,
    num_sess_multi,
    num_trials_per_sess_multi,
    true_glmhmm_multi,
):
    true_latents_multi, true_choices_multi = [], []
    for sess_idx_multi in range(num_sess_multi):
        true_z, true_y = true_glmhmm_multi.sample(
            num_trials_per_sess_multi, input=inpts_multi[sess_idx_multi]
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
    inpts_multi,
    input_dim_multi,
    num_categories_multi,
    num_states_multi,
    obs_dim_multi,
    ssm,
    true_choices_multi,
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
def _(
    find_permutation,
    inpts_multi,
    new_glmhmm_multi,
    true_choices_multi,
    true_latents_multi,
):
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
    new_glmhmm_multi,
    np,
    num_categories_multi,
    num_states_multi,
    plt,
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

    for class_idx_multiplot in range(num_categories_multi):
        plt.subplot(2, num_categories_multi + 1, class_idx_multiplot + 1)
        if class_idx_multiplot < num_categories_multi - 1:
            for state_idx_multiplot in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    gen_weights_multi[state_idx_multiplot, class_idx_multiplot],
                    marker="o",
                    color=cols_multi[state_idx_multiplot],
                    lw=1.5,
                    label=f"state {state_idx_multiplot + 1}; class {class_idx_multiplot + 1}",
                )
        else:
            for state_idx_multiplot in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    np.zeros(input_dim_multi),
                    marker="o",
                    color=cols_multi[state_idx_multiplot],
                    lw=1.5,
                    label=f"state {state_idx_multiplot + 1}; class {class_idx_multiplot + 1}",
                    alpha=0.5,
                )

        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.yticks(fontsize=10)
        plt.xticks([0, 1], ["", ""])
        if class_idx_multiplot == 0:
            plt.ylabel("GLM weight", fontsize=15)
        plt.legend()
        plt.title("Generative weights; class " + str(class_idx_multiplot + 1), fontsize=15)
        plt.ylim((-3, 10))

    plt.subplot(2, num_categories_multi + 1, num_categories_multi + 1)
    gen_trans_mat = np.exp(gen_log_trans_mat_multi)[0]
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx_multitrans in range(gen_trans_mat.shape[0]):
        for col_idx_multitrans in range(gen_trans_mat.shape[1]):
            plt.text(
                col_idx_multitrans,
                row_idx_multitrans,
                str(np.around(gen_trans_mat[row_idx_multitrans, col_idx_multitrans], decimals=2)),
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

    for class_idx_multiplot in range(num_categories_multi):
        plt.subplot(2, num_categories_multi + 1, num_categories_multi + class_idx_multiplot + 2)
        if class_idx_multiplot < num_categories_multi - 1:
            for state_idx_multiplot in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    recovered_weights[state_idx_multiplot, class_idx_multiplot],
                    marker="o",
                    linestyle="--",
                    color=cols_multi[state_idx_multiplot],
                    lw=1.5,
                    label=f"state {state_idx_multiplot + 1}; class {class_idx_multiplot + 1}",
                )
        else:
            for state_idx_multiplot in range(num_states_multi):
                plt.plot(
                    range(input_dim_multi),
                    np.zeros(input_dim_multi),
                    marker="o",
                    linestyle="--",
                    color=cols_multi[state_idx_multiplot],
                    lw=1.5,
                    label=f"state {state_idx_multiplot + 1}; class {class_idx_multiplot + 1}",
                    alpha=0.5,
                )

        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.yticks(fontsize=10)
        plt.xlabel("covariate", fontsize=15)
        if class_idx_multiplot == 0:
            plt.ylabel("GLM weight", fontsize=15)
        plt.xticks([0, 1], ["stimulus", "bias"], fontsize=12, rotation=45)
        plt.legend()
        plt.title("Recovered weights; class " + str(class_idx_multiplot + 1), fontsize=15)
        plt.ylim((-3, 10))

    plt.subplot(2, num_categories_multi + 1, 2 * num_categories_multi + 2)
    recovered_trans_mat = np.exp(recovered_transitions)[0]
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap="bone")
    for row_idx_multitrans in range(recovered_trans_mat.shape[0]):
        for col_idx_multitrans in range(recovered_trans_mat.shape[1]):
            plt.text(
                col_idx_multitrans,
                row_idx_multitrans,
                str(np.around(recovered_trans_mat[row_idx_multitrans, col_idx_multitrans], decimals=2)),
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
    return (gen_trans_mat,)


if __name__ == "__main__":
    app.run()
