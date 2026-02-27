import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LM-HMM Tutorial Notebook

    This notebook walks through synthetic LM-HMM generation, parameter recovery
    with MLE/MAP, and optional model-selection analyses.
    """)
    return


@app.cell
def _():
    import sys, os
    import itertools
    from pathlib import Path

    import matplotlib.pyplot as plt
    import autograd.numpy as np
    import autograd.numpy.random as npr
    import multiprocessing
    import seaborn as sns

    from joblib import Parallel, delayed
    from matplotlib.patches import Patch
    from scipy.interpolate import interp1d
    from sklearn.model_selection import StratifiedKFold

    npr.seed(3)

    sns.set_style("white")
    sns.set_context("talk")

    try:
        import ssm
        from ssm.plots import gradient_cmap, white_to_color_cmap
        from ssm.util import find_permutation
    except ImportError as exc:
        raise ImportError(
            "The ssm package is required. Install the fork used by this tutorial."
        ) from exc

    try:
        from ssm import utilplot
    except Exception:
        import utilplot

    fig_prefix = "LM-HMM"
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
        gradient_cmap,
        interp1d,
        multiprocessing,
        np,
        plt,
        sns,
        ssm,
        utilplot,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting Palette

    Define the color palette and gradient colormap used consistently across all
    diagnostic plots in the notebook.
    """)
    return


@app.cell
def _(gradient_cmap, sns):
    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange",
    ]
    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)
    return cmap, colors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Runtime Controls

    Configure which expensive sections run, along with model-selection
    hyperparameters and parallel worker settings.
    """)
    return


@app.cell
def _(multiprocessing):
    # Runtime controls for expensive sections.
    save_figures = True
    run_map = True
    run_model_selection = True

    # Model-selection defaults.
    num_sess = 3
    max_states = 4
    n_iters_cv = 1000
    tol_cv = 1e-4
    n_run_em = 4
    num_threads = max(1, (multiprocessing.cpu_count() - 1) * 2)
    return (
        max_states,
        n_iters_cv,
        n_run_em,
        num_sess,
        num_threads,
        run_map,
        run_model_selection,
        save_figures,
        tol_cv,
    )


@app.cell(hide_code=True)
def _(
    max_states,
    mo,
    n_iters_cv,
    n_run_em,
    num_sess,
    num_threads,
    run_map,
    run_model_selection,
    save_figures,
    tol_cv,
):
    mo.md(rf"""
    **Run Controls**

    - `save_figures = {save_figures}`
    - `run_map = {run_map}`
    - `run_model_selection = {run_model_selection}`
    - `num_threads = {num_threads}`

    Model-selection parameters:
    - `num_sess = {num_sess}`
    - `max_states = {max_states}`
    - `n_iters_cv = {n_iters_cv}`
    - `tol_cv = {tol_cv}`
    - `n_run_em = {n_run_em}`

    Edit the control cell above to change behavior.
    """)
    return


@app.cell
def _(configure_figure_saving, save_figures):
    configure_figure_saving(save_figures)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Generate LM-HMM observations

    ### 1a. Specify parameters of the ground truth LM-HMM

    Here we generate observations from an LM-HMM `true_hmm` with `input_dim` dimensional
    inputs, `obs_dim` dimensional observations, and `num_states` hidden states.
    """)
    return


@app.cell
def _(np, ssm):
    # Set the parameters of the HMM
    time_bins = 500
    num_states = 3
    obs_dim = 1
    input_dim = 1

    true_hmm = ssm.HMM(
        num_states,
        obs_dim,
        M=input_dim,
        observations="input_driven_obs_gaussian",
        transitions="standard",
    )

    gen_weights = np.random.randn(num_states, obs_dim, input_dim)
    mus = np.random.randn(num_states, obs_dim)

    stdnoise = 1
    sigma = stdnoise**2 * np.eye(obs_dim)
    sigmas = np.dstack([sigma] * num_states).transpose((2, 0, 1))

    true_hmm.observations.mus = mus
    true_hmm.observations.Sigmas = sigmas
    true_hmm.observations.Wks = 2 * gen_weights

    trans_eps = 0.02
    trans0 = trans_eps * np.ones((num_states, num_states))
    for i in range(num_states):
        trans0[i, i] = 1 - (trans_eps * (num_states - 1))
    true_hmm.transitions.log_Ps = np.log(trans0)

    print("trans\n", trans0)
    print("mus\n", mus)
    print("Wks\n", gen_weights)
    print("Sigmas\n", sigmas)
    return input_dim, num_states, obs_dim, time_bins, trans0, true_hmm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1a.i Visualize Generative Parameters

    Plot the true observation model parameters and transition matrix that will be
    used to synthesize data.
    """)
    return


@app.cell
def _(plt, trans0, true_hmm, utilplot):
    # Plot generative parameters.
    true_mus = true_hmm.observations.mus
    true_weights = true_hmm.observations.Wks

    utilplot.plot_weights(true_weights, true_mus)

    plt.figure(figsize=(5, 5), dpi=80, facecolor="w", edgecolor="k")
    utilplot.plot_trans_matrix(trans0)
    plt.tight_layout()
    plt.show()
    return true_mus, true_weights


@app.cell
def _(mo):
    mo.md(r"""
    ### 1b. Sample data from the LM-HMM

    We generate a random exogenous input of shape `(time_bins, input_dim)` and sample
    latent states and observations from the generative LM-HMM.
    """)
    return


@app.cell
def _(input_dim, np, time_bins, true_hmm):
    inpt = np.random.rand(time_bins)
    if inpt.ndim == 1:
        inpt = np.expand_dims(inpt, axis=1)
    inpt = np.tile(inpt, input_dim)

    true_states, obs = true_hmm.sample(time_bins, input=inpt)
    true_ll = true_hmm.log_likelihood(obs, inputs=inpt)
    return inpt, obs, true_ll, true_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1b.i Scatter Plot by Latent State

    Show observations in input-observation space, colored by true latent state,
    to make state-dependent emission structure visible.
    """)
    return


@app.cell
def _(colors, inpt, num_states, obs, plt, true_states):
    plt.figure(figsize=(6, 6))
    for k in range(num_states):
        plt.plot(
            inpt[true_states == k, 0],
            obs[true_states == k, 0],
            "o",
            mfc=colors[k],
            mec="none",
            ms=4,
        )

    plt.plot(inpt[:, 0], obs[:, 0], "-k", lw=1, alpha=0.25)
    plt.xlabel("input $x_1$")
    plt.ylabel("obs $y_1$")
    plt.title("Observation Distributions")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1b.ii Posterior Ribbon for Ground Truth States

    Build one-hot posterior probabilities from the known latent sequence and plot
    them against observations.
    """)
    return


@app.cell
def _(cmap, colors, inpt, np, obs, true_hmm, true_states, utilplot):
    posterior_probs0 = np.zeros((true_states.size, true_states.max() + 1))
    posterior_probs0[np.arange(true_states.size), true_states] = 1

    utilplot.plot_postprob_obs(posterior_probs0, obs, inpt, true_hmm, colors, cmap)
    return (posterior_probs0,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Fit LM-HMM and perform recovery analysis

    ### 2a. Maximum Likelihood Estimation (MLE)
    """)
    return


@app.cell
def _(inpt, np, num_states, obs, obs_dim, plt, ssm, true_ll):
    mle_hmm = ssm.HMM(
        num_states,
        obs_dim,
        M=inpt.shape[1],
        observations="input_driven_obs_gaussian",
        transitions="standard",
    )

    n_iters_mle = 10000
    tol_mle = 1e-6
    fit_ll = mle_hmm.fit(obs, inputs=inpt, method="em", num_iters=n_iters_mle, tolerance=tol_mle)

    plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(fit_ll, label="EM")
    plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ":k", label="True")
    plt.legend(loc="lower right")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Probability")
    plt.show()
    return (mle_hmm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a.i Align Recovered States

    Resolve label switching by permuting inferred MLE states to best match the
    ground-truth latent sequence.
    """)
    return


@app.cell
def _(find_permutation, inpt, mle_hmm, obs, true_states):
    most_likely_states = mle_hmm.most_likely_states(obs, input=inpt)
    mle_hmm.permute(find_permutation(true_states, most_likely_states))
    print("State permutation aligned to ground truth ordering.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a.ii Compare Emission Parameters

    Compare true and recovered observation weights and means after state alignment.
    """)
    return


@app.cell
def _(mle_hmm, true_mus, true_weights, utilplot):
    recovered_weights = mle_hmm.observations.Wks
    recovered_mus = mle_hmm.observations.mus

    weight_dic = {
        0: {"weights": true_weights, "mus": true_mus, "label": "true"},
        1: {"weights": recovered_weights, "mus": recovered_mus, "label": "mle"},
    }
    utilplot.plot_weights_comparison(weight_dic)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a.iii Compare Transition Matrices

    Visualize the true transition matrix alongside the matrix recovered by MLE.
    """)
    return


@app.cell
def _(mle_hmm, np, plt, true_hmm, utilplot):
    gen_trans_mat = np.exp(true_hmm.transitions.log_Ps)
    recovered_trans_mat = np.exp(mle_hmm.transitions.log_Ps)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    utilplot.plot_trans_matrix(gen_trans_mat)
    plt.title("Generative transition matrix", fontsize=15)

    plt.subplot(1, 2, 2)
    utilplot.plot_trans_matrix(recovered_trans_mat)
    plt.title("Recovered transition matrix", fontsize=15)

    utilplot.plt.subplots_adjust(0, 0, 1, 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a.iv Compare Posterior State Probabilities

    Plot posterior state probabilities from the fitted MLE model against the
    ground-truth one-hot posteriors.
    """)
    return


@app.cell
def _(cmap, colors, inpt, mle_hmm, obs, posterior_probs0, true_hmm, utilplot):
    posterior_probs = mle_hmm.expected_states(data=obs, input=inpt)[0]

    print("true")
    utilplot.plot_postprob_obs(posterior_probs0, obs, inpt, true_hmm, colors, cmap)
    print("mle")
    utilplot.plot_postprob_obs(posterior_probs, obs, inpt, mle_hmm, colors, cmap)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a.v Duration Diagnostics

    Compare true versus inferred run-length distributions for each latent state.
    """)
    return


@app.cell
def _(colors, inpt, mle_hmm, num_states, obs, plt, ssm, true_states):
    hmm_z = mle_hmm.most_likely_states(obs, input=inpt)
    true_state_list, true_durations = ssm.util.rle(true_states)
    inferred_state_list, inferred_durations = ssm.util.rle(hmm_z)

    true_durs_stacked = []
    inf_durs_stacked = []
    for s in range(num_states):
        true_durs_stacked.append(true_durations[true_state_list == s])
        inf_durs_stacked.append(inferred_durations[inferred_state_list == s])

    plt.figure(figsize=(8, 4))
    plt.hist(
        true_durs_stacked,
        label=["state " + str(s) for s in range(num_states)],
        color=colors[:num_states],
    )
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of True State Durations")

    plt.figure(figsize=(8, 4))
    plt.hist(
        inf_durs_stacked,
        label=["state " + str(s) for s in range(num_states)],
        color=colors[:num_states],
    )
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of Inferred State Durations")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2d. Maximum A Posteriori (MAP) estimation

    Set `run_map = True` in the control cell to run this section.
    """)
    return


@app.cell
def _(
    find_permutation,
    inpt,
    np,
    num_states,
    obs,
    obs_dim,
    plt,
    run_map,
    ssm,
    true_ll,
    true_states,
):
    if run_map:
        prior_sigma = 1
        prior_alpha = 1

        map_hmm = ssm.HMM(
            num_states,
            obs_dim,
            M=inpt.shape[1],
            observations="input_driven_obs_gaussian",
            observation_kwargs=dict(prior_sigma=prior_sigma),
            transitions="sticky",
            transition_kwargs=dict(alpha=prior_alpha, kappa=0),
        )

        n_iters_map = 10000
        tol_map = 1e-6
        fit_ll_map = map_hmm.fit(
            obs,
            inputs=inpt,
            method="em",
            num_iters=n_iters_map,
            tolerance=tol_map,
        )

        most_likely_states_map = map_hmm.most_likely_states(obs, input=inpt)
        map_hmm.permute(find_permutation(true_states, most_likely_states_map))

        plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
        plt.plot(fit_ll_map, label="EM")
        plt.plot([0, len(fit_ll_map)], true_ll * np.ones(2), ":k", label="True")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.show()
    else:
        map_hmm = None
        fit_ll_map = None
        most_likely_states_map = None
        print("MAP section skipped. Set run_map = True to enable.")
    return (map_hmm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2d.i Compare Final Log-Likelihoods

    Summarize final data likelihood under true, MLE, and MAP models.
    """)
    return


@app.cell
def _(inpt, map_hmm, mle_hmm, obs, plt, true_hmm):
    true_likelihood = true_hmm.log_likelihood(obs, inputs=inpt)
    mle_final_ll = mle_hmm.log_likelihood(obs, inputs=inpt)
    map_final_ll = map_hmm.log_likelihood(obs, inputs=inpt) if map_hmm is not None else float("nan")

    plt.figure(figsize=(2.5, 3), dpi=80, facecolor="w", edgecolor="k")
    loglikelihood_vals = [true_likelihood, mle_final_ll, map_final_ll]
    colors_ll = ["red", "navy", "purple"]
    labels = ["true", "mle", "map"]

    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width=0.8, color=colors_ll[z])

    plt.xticks([0, 1, 2], labels, fontsize=10)
    plt.xlabel("model", fontsize=12)
    plt.ylabel("loglikelihood", fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2d.ii Compare Emission Recovery Across Estimators

    Extend the parameter comparison to include MAP estimates when available.
    """)
    return


@app.cell
def _(map_hmm, mle_hmm, true_mus, true_weights, utilplot):
    recovered_weights_mle = mle_hmm.observations.Wks
    recovered_mus_mle = mle_hmm.observations.mus

    weight_dic = {
        0: {"weights": true_weights, "mus": true_mus, "label": "true"},
        1: {"weights": recovered_weights_mle, "mus": recovered_mus_mle, "label": "mle"},
    }

    if map_hmm is not None:
        recovered_weights_map = map_hmm.observations.Wks
        recovered_mus_map = map_hmm.observations.mus
        weight_dic[2] = {
            "weights": recovered_weights_map,
            "mus": recovered_mus_map,
            "label": "map",
        }

    utilplot.plot_weights_comparison(weight_dic)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Model Selection

    Set `run_model_selection = True` in the control cell to run cross-validation and
    information-criterion calculations.
    """)
    return


@app.cell
def _(inpt, np, num_sess, true_hmm):
    # Generate multiple sessions.
    time_bins_cv = len(inpt)
    inputs = []
    output = []
    true_latents = []

    for sess_idx_copy in range(num_sess):
        inputs.append(inpt)
    for sess in range(num_sess):
        true_z, true_y = true_hmm.sample(time_bins_cv, input=inputs[sess])
        true_latents.append(true_z)
        output.append(true_y)

    inputs0 = np.vstack(inputs)
    output0 = np.vstack(output)

    ylabel_mouse = [idx * np.ones(len(bout_vec_copy)) for idx, bout_vec_copy in enumerate(output)]
    ylabel_mouse = np.hstack(ylabel_mouse)
    return inputs0, output0, ylabel_mouse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3a. Visualize Stratified CV Splits

    Construct and plot fold assignments used for cross-validation.
    """)
    return


@app.cell
def _(
    Patch,
    StratifiedKFold,
    output0,
    plt,
    save_figures,
    utilplot,
    ylabel_mouse,
):
    ntrials = len(output0)
    nKfold = min(ntrials, 4)
    synthetic_data = output0

    fig, ax = plt.subplots(figsize=(6, 3))
    plotlabels = {"class": "equal", "group": "Mouse", "x": "Bouts"}
    utilplot.plot_cv_indices(
        StratifiedKFold(nKfold),
        synthetic_data,
        ylabel_mouse,
        ylabel_mouse,
        ax,
        nKfold,
        plotlabels=plotlabels,
    )

    cmap_cv = plt.cm.coolwarm
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)

    plt.show()
    return nKfold, synthetic_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3b. Extract Dimensions for CV Models

    Infer input and observation dimensions from stacked synthetic sessions.
    """)
    return


@app.cell
def _(inputs0, synthetic_data):
    synthetic_inpts = inputs0
    obs_dim_cv = len(synthetic_data[0])
    input_dim_cv = len(synthetic_inpts[0])
    return input_dim_cv, obs_dim_cv, synthetic_inpts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c. Define Fold-Level Training Function

    Implement a helper that fits MLE and MAP models on one training split and
    returns per-trial train/held-out log-likelihoods.
    """)
    return


@app.cell
def _(ssm):
    def xval_func(data_in, num_states0):
        training_data = data_in["training_data"]
        test_data = data_in["test_data"]
        training_inpts = data_in["training_inpts"]
        test_inpts = data_in["test_inpts"]
        n_iters = data_in["N_iters"]
        tol = data_in["TOL"]

        obs_dim_local = len(training_data[0])
        input_dim_local = len(training_inpts[0])
        n_train = len(training_data)
        n_test = len(test_data)

        out = {}

        mle_hmm_local = ssm.HMM(
            num_states0,
            obs_dim_local,
            M=input_dim_local,
            observations="input_driven_obs_gaussian",
            transitions="standard",
        )
        mle_hmm_local.fit(
            training_data,
            inputs=training_inpts,
            method="em",
            num_iters=n_iters,
            tolerance=tol,
        )
        out["ll_training"] = (
            mle_hmm_local.log_likelihood(training_data, inputs=training_inpts) / n_train
        )
        out["ll_heldout"] = mle_hmm_local.log_likelihood(test_data, inputs=test_inpts) / n_test

        prior_sigma = 2
        prior_alpha = 2
        map_hmm_local = ssm.HMM(
            num_states0,
            obs_dim_local,
            M=input_dim_local,
            observations="input_driven_obs_gaussian",
            observation_kwargs=dict(prior_sigma=prior_sigma),
            transitions="sticky",
            transition_kwargs=dict(alpha=prior_alpha, kappa=0),
        )
        map_hmm_local.fit(
            training_data,
            inputs=training_inpts,
            method="em",
            num_iters=n_iters,
            tolerance=tol,
        )
        out["ll_training_map"] = (
            map_hmm_local.log_likelihood(training_data, inputs=training_inpts) / n_train
        )
        out["ll_heldout_map"] = map_hmm_local.log_likelihood(test_data, inputs=test_inpts) / n_test

        return out

    return (xval_func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d. Run Cross-Validation Grid

    Sweep hidden-state counts and random initializations across folds in
    parallel, collecting train and held-out scores for MLE and MAP.
    """)
    return


@app.cell
def _(
    Parallel,
    delayed,
    max_states,
    nKfold,
    n_iters_cv,
    n_run_em,
    np,
    num_threads,
    run_model_selection,
    synthetic_data,
    synthetic_inpts,
    tol_cv,
    xval_func,
    ylabel_mouse,
):
    if run_model_selection:
        ll_training = np.zeros((max_states, nKfold, n_run_em))
        ll_heldout = np.zeros((max_states, nKfold, n_run_em))
        ll_training_map = np.zeros((max_states, nKfold, n_run_em))
        ll_heldout_map = np.zeros((max_states, nKfold, n_run_em))

        stN = np.flip(np.tile(np.arange(1, max_states + 1), n_run_em))
        runN = np.repeat(np.arange(1, n_run_em + 1), max_states, axis=0)

        print(f"Running parallel code with {num_threads} workers")

        from sklearn.model_selection import StratifiedKFold as _SKF

        skf = _SKF(n_splits=nKfold)
        for iK, (train_index, test_index) in enumerate(skf.split(synthetic_data, ylabel_mouse)):
            data_in = {
                "training_data": synthetic_data[train_index],
                "test_data": synthetic_data[test_index],
                "training_inpts": synthetic_inpts[train_index],
                "test_inpts": synthetic_inpts[test_index],
                "N_iters": n_iters_cv,
                "TOL": tol_cv,
            }

            results = Parallel(n_jobs=num_threads)(
                delayed(xval_func)(data_in, num_states0)
                for _iRun, num_states0 in zip(runN, stN)
            )

            for _i in range(max_states * n_run_em):
                ll_training[stN[_i] - 1, iK, runN[_i] - 1] = results[_i]["ll_training"]
                ll_heldout[stN[_i] - 1, iK, runN[_i] - 1] = results[_i]["ll_heldout"]
                ll_training_map[stN[_i] - 1, iK, runN[_i] - 1] = results[_i]["ll_training_map"]
                ll_heldout_map[stN[_i] - 1, iK, runN[_i] - 1] = results[_i]["ll_heldout_map"]
        ran_xval = True
    else:
        ll_training = None
        ll_heldout = None
        ll_training_map = None
        ll_heldout_map = None
        ran_xval = False
        print("Cross-validation skipped. Set run_model_selection = True to enable.")
    return ll_heldout, ll_heldout_map, ll_training, ll_training_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3e. Compute Information Criteria

    Fit full-data models across candidate state counts and compute AIC/BIC
    distributions over random restarts.
    """)
    return


@app.cell
def _(
    Parallel,
    delayed,
    input_dim_cv,
    max_states,
    n_iters_cv,
    n_run_em,
    np,
    obs_dim_cv,
    run_model_selection,
    ssm,
    synthetic_data,
    synthetic_inpts,
    tol_cv,
):
    if run_model_selection:
        def single_func(synthetic_data_local, synthetic_inpts_local, num_states_local):
            xval_hmm = ssm.HMM(
                num_states_local,
                obs_dim_cv,
                M=input_dim_cv,
                observations="input_driven_obs_gaussian",
                transitions="standard",
            )
            xval_hmm.fit(
                synthetic_data_local,
                inputs=synthetic_inpts_local,
                method="em",
                num_iters=n_iters_cv,
                tolerance=tol_cv,
            )
            return xval_hmm.log_likelihood(synthetic_data_local, inputs=synthetic_inpts_local)

        time_bins_all = len(synthetic_data)
        BIC = np.zeros((max_states, n_run_em))
        AIC = np.zeros((max_states, n_run_em))

        for iS, num_states_local in enumerate(range(1, max_states + 1)):
            K = (num_states_local + 1) * (num_states_local - 1) + num_states_local * (
                obs_dim_cv * input_dim_cv + 2 * obs_dim_cv
            )

            _results = Parallel(n_jobs=1)(
                delayed(single_func)(synthetic_data, synthetic_inpts, num_states_local)
                for run_idx_ic in range(n_run_em)
            )

            for iRun in range(n_run_em):
                BIC[iS, iRun] = K * np.log(time_bins_all) - 2 * _results[iRun]
                AIC[iS, iRun] = K * 2 - 2 * _results[iRun]
        ran_ic = True
    else:
        AIC = None
        BIC = None
        ran_ic = False
        print("BIC/AIC section skipped. Set run_model_selection = True to enable.")
    return AIC, BIC, iS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3f. Summarize Model-Selection Results

    Plot cross-validation performance and information criteria versus state count.
    """)
    return


@app.cell
def _(
    AIC,
    BIC,
    iS,
    interp1d,
    ll_heldout,
    ll_heldout_map,
    ll_training,
    ll_training_map,
    max_states,
    nKfold,
    n_run_em,
    np,
    plt,
    run_model_selection,
    save_figures,
):
    _ = interp1d  # kept for parity with original notebook imports.

    if run_model_selection and all(
        metric_item_cv is not None
        for metric_item_cv in [ll_training, ll_heldout, ll_training_map, ll_heldout_map, AIC, BIC]
    ):
        plt.figure(figsize=(20, 10), dpi=80, facecolor="w", edgecolor="k")

        plt.subplot(1, 2, 1)

        ll_training_plot = ll_training.reshape(max_states, nKfold * n_run_em)
        ll_heldout_plot = ll_heldout.reshape(max_states, nKfold * n_run_em)
        ll_training_map_plot = ll_training_map.reshape(max_states, nKfold * n_run_em)
        ll_heldout_map_plot = ll_heldout_map.reshape(max_states, nKfold * n_run_em)

        for _iS in range(max_states):
            plt.plot(
                (_iS + 1) * np.ones(nKfold * n_run_em),
                ll_training_plot[iS, :],
                color="tab:blue",
                marker="o",
                lw=0,
            )
            plt.plot(
                (_iS + 1) * np.ones(nKfold * n_run_em),
                ll_heldout_plot[iS, :],
                color="tab:orange",
                marker="o",
                lw=0,
            )
            plt.plot(
                (_iS + 1) * np.ones(nKfold * n_run_em),
                ll_training_map_plot[iS, :],
                color="tab:green",
                marker="o",
                lw=0,
            )
            plt.plot(
                (_iS + 1) * np.ones(nKfold * n_run_em),
                ll_heldout_map_plot[iS, :],
                color="tab:red",
                marker="o",
                lw=0,
            )

        x = range(1, max_states + 1)
        y = ll_training_plot.mean(axis=1)
        error = ll_training_plot.std(axis=1)
        plt.plot(x, y, label="training_MLE", color="tab:blue")
        plt.fill_between(x, y - error, y + error, alpha=0.1)

        y = ll_heldout_plot.mean(axis=1)
        error = ll_heldout_plot.std(axis=1)
        plt.plot(x, y, label="test_MLE", color="tab:orange")
        plt.fill_between(x, y - error, y + error, alpha=0.1)

        y = ll_training_map_plot.mean(axis=1)
        error = ll_training_map_plot.std(axis=1)
        plt.plot(x, y, label="training_MAP", color="tab:green")
        plt.fill_between(x, y - error, y + error, alpha=0.1)

        y = ll_heldout_map_plot.mean(axis=1)
        error = ll_heldout_map_plot.std(axis=1)
        plt.plot(x, y, label="test_MAP", color="tab:red")
        plt.fill_between(x, y - error, y + error, alpha=0.1)

        plt.legend(loc="lower right")
        plt.xlabel("states")
        plt.xlim(0, max_states + 1)
        plt.ylabel("Log-Likelihood per trial")

        plt.subplot(1, 2, 2)
        x = range(1, max_states + 1)

        y = np.mean(BIC, 1)
        error = np.std(BIC, 1)
        plt.plot(x, y, label="BIC")
        plt.fill_between(
            x,
            y - error,
            y + error,
            alpha=0.5,
            edgecolor="#CC4F1B",
            facecolor="#FF9848",
        )

        y = np.mean(AIC, 1)
        error = np.std(AIC, 1)
        plt.plot(x, y, label="AIC")
        plt.xlabel("states")
        plt.xlim(0, max_states + 1)
        plt.ylabel("criterion")
        plt.legend(loc="lower left")

        plt.show()

        model_sel = {
            "ll_training": ll_training,
            "ll_heldout": ll_heldout,
            "ll_training_map": ll_training_map,
            "ll_heldout_map": ll_heldout_map,
            "BIC": BIC,
            "AIC": AIC,
        }
    else:
        model_sel = {}
    return error, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3g. Criterion-Only Quick View

    Generate a compact AIC/BIC-only diagnostic plot for rapid comparison.
    """)
    return


@app.cell
def _(AIC, BIC, error, max_states, np, plt, run_model_selection, x, y):
    if run_model_selection and BIC is not None and AIC is not None:
        _x = range(1, max_states + 1)

        _y = np.mean(BIC, 1)
        _error = np.std(BIC, 1)
        plt.plot(_x, _y, label="BIC")
        plt.fill_between(
            _x,
            _y - error,
            _y + error,
            alpha=0.5,
            edgecolor="#CC4F1B",
            facecolor="#FF9848",
        )

        _y = np.mean(AIC, 1)
        _error = np.std(AIC, 1)
        plt.plot(x, y, label="AIC")
        plt.xlabel("states")
        plt.xlim(0, max_states + 1)
        plt.ylabel("criterion")
        plt.legend(loc="lower left")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
