get_test_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    stop("Could not determine test script path from commandArgs().")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

# Build block rows with current CSV conventions for residual-model tests.
#
# Returns:
#   data.frame with shape (6 blocks, 4 columns). `block_type` alternates right
#   and left labels and all numeric values are stored as character strings to
#   match CSV loading behavior.
make_block_model_df <- function() {
  data.frame(
    block_ix = as.character(0:5),
    block_type = c(
      "right_uncued", "left_uncued", "right_cued",
      "left_cued", "right_uncued", "left_uncued"
    ),
    prev_n_rewarded = as.character(c(0, 1, 2, 3, 4, 5)),
    trials_to_correct = as.character(c(1, 3, 2, 5, 3, 7)),
    stringsAsFactors = FALSE
  )
}

test_script_path <- get_test_script_path()
rcode_root <- normalizePath(
  file.path(dirname(test_script_path), ".."),
  winslash = "/",
  mustWork = TRUE
)

source(file.path(rcode_root, "preprocessing", "block_preprocessing.R"))
source(file.path(rcode_root, "block_residual_modeling.R"))

block_df <- clean_block_dataframe(make_block_model_df())
block_side <- derive_block_side(block_df$block_type)
stopifnot(identical(as.character(block_side), c("right", "left", "right", "left", "right", "left")))
stopifnot(identical(levels(block_side), c("right", "left")))

prepared <- prepare_block_residual_model_dataframe(block_df)
stopifnot(identical(
  colnames(prepared$x),
  c("prev_n_rewarded", "block_sideleft", "prev_n_rewarded:block_sideleft")
))
stopifnot(identical(as.numeric(prepared$x[2, ]), c(1, 1, 1)))
stopifnot(identical(as.numeric(prepared$x[3, ]), c(2, 0, 0)))
stopifnot(identical(prepared$valid_rows, 1:6))

metrics <- compute_residual_spread_metrics(c(-2, -1, 0, 1, 2))
stopifnot(isTRUE(all.equal(metrics$residual_mad_raw, 1)))
stopifnot(isTRUE(all.equal(metrics$residual_mad_scaled, 1.4826)))
stopifnot(isTRUE(all.equal(metrics$residual_iqr, 2)))
stopifnot(isTRUE(all.equal(metrics$residual_rmse, sqrt(2))))
stopifnot(isTRUE(all.equal(metrics$residual_sd, stats::sd(c(-2, -1, 0, 1, 2)))))

single_metric <- compute_residual_spread_metrics(c(1))
stopifnot(identical(single_metric$residual_sd, BLOCK_RESIDUAL_MISSING_VALUE))

fit_result <- add_block_residual_model_outputs(block_df, seed = 123, min_valid_blocks = 5)
stopifnot(all(c(
  "lasso_lambda_min_residual_TTS",
  "lasso_lambda_1se_residual_TTS",
  "elastic_net_lambda_min_residual_TTS",
  "elastic_net_lambda_1se_residual_TTS",
  "lasso_rewards_x_side_lambda_min_residual_TTS",
  "lasso_rewards_x_side_lambda_1se_residual_TTS",
  "elastic_net_rewards_x_side_lambda_min_residual_TTS",
  "elastic_net_rewards_x_side_lambda_1se_residual_TTS",
  "lasso_rewards_plus_side_lambda_min_residual_TTS",
  "lasso_rewards_plus_side_lambda_1se_residual_TTS",
  "elastic_net_rewards_plus_side_lambda_min_residual_TTS",
  "elastic_net_rewards_plus_side_lambda_1se_residual_TTS"
) %in% names(fit_result$block_df)))
stopifnot(nrow(fit_result$summary_df) == 8)
stopifnot(all(c(
  "model_formula",
  "coefficient_intercept",
  "coefficient_prev_n_rewarded",
  "coefficient_block_side_left",
  "coefficient_prev_n_rewarded_block_side_left",
  "right_intercept",
  "left_intercept",
  "average_intercept",
  "right_reward_slope",
  "left_reward_slope",
  "average_reward_slope",
  "side_intercept_delta_left_minus_right",
  "side_reward_slope_delta_left_minus_right",
  "residual_sd"
) %in% names(fit_result$summary_df)))
stopifnot(isTRUE(all.equal(
  as.numeric(fit_result$summary_df$average_intercept),
  (
    as.numeric(fit_result$summary_df$right_intercept) +
      as.numeric(fit_result$summary_df$left_intercept)
  ) / 2
)))
stopifnot(isTRUE(all.equal(
  as.numeric(fit_result$summary_df$average_reward_slope),
  (
    as.numeric(fit_result$summary_df$right_reward_slope) +
      as.numeric(fit_result$summary_df$left_reward_slope)
  ) / 2
)))
stopifnot(identical(
  sort(unique(fit_result$summary_df$model_formula)),
  c("rewards_plus_side", "rewards_x_side")
))
additive_rows <- fit_result$summary_df[fit_result$summary_df$model_formula == "rewards_plus_side", ]
stopifnot(all(additive_rows$coefficient_prev_n_rewarded_block_side_left == 0))
stopifnot(isTRUE(all.equal(
  as.numeric(additive_rows$right_reward_slope),
  as.numeric(additive_rows$left_reward_slope)
)))

small_result <- add_block_residual_model_outputs(block_df[1:4, ], seed = 123, min_valid_blocks = 5)
stopifnot(all(small_result$block_df$lasso_lambda_min_residual_TTS == "None"))
stopifnot(all(small_result$summary_df$coefficient_intercept == "None"))
stopifnot(all(small_result$summary_df$residual_sd == "None"))
stopifnot(identical(small_result$summary_df$n_valid_blocks, rep(4L, 8)))

# Verify explicit opt-in for a side-free model without changing old defaults.
# Input: no arguments. Return: invisible NULL; coefficients retain trial units.
test_rewards_only_preparation <- function() {
  rewards_only <- prepare_block_residual_model_dataframe(block_df, "rewards_only")
  stopifnot(identical(dim(rewards_only$x), c(6L, 1L)))
  stopifnot(identical(colnames(rewards_only$x), "prev_n_rewarded"))
  stopifnot(identical(as.numeric(rewards_only$x), 0:5 + 0))
  stopifnot(identical(vapply(BLOCK_RESIDUAL_FORMULA_SPECS, `[[`, "", "model_formula"),
                      c("rewards_x_side", "rewards_plus_side")))
  invisible(NULL)
}

# Verify an inert adapter preserves the one-predictor elastic-net objective.
# Input: no arguments. Return: invisible NULL. Both x/y have population SD 1,
# making the Gaussian glmnet objective directly comparable to soft thresholding.
test_one_predictor_elastic_net_objective <- function() {
  rewards <- seq_len(30)
  x <- as.numeric(rewards - mean(rewards))
  x <- x / sqrt(mean(x^2))
  y <- 0.8 * x + sin(rewards)
  y <- (y - mean(y)) / sqrt(mean((y - mean(y))^2))
  penalties <- c(0.5, 0.2, 0.1)
  alpha <- 0.5
  folds <- rep(seq_len(5), length.out = length(y))
  for (direction in c(-1, 1)) {
    response <- direction * y
    result <- fit_block_glmnet_cv(
      matrix(x, ncol = 1, dimnames = list(NULL, "reward")), response,
      alpha = alpha, fold_id = folds, lambda = penalties
    )
    coefficients <- as.matrix(stats::coef(result$fit, s = penalties))
    correlation <- mean(x * response)
    expected <- sign(correlation) * pmax(abs(correlation) - penalties * alpha, 0) /
      (mean(x^2) + penalties * (1 - alpha))
    stopifnot(isTRUE(all.equal(as.numeric(coefficients["reward", ]), expected,
                               tolerance = 1e-6)))
    stopifnot(all(abs(coefficients["(Intercept)", ]) < 1e-10))
    predictions <- as.matrix(stats::predict(result$fit, newx = result$x, s = penalties))
    stopifnot(isTRUE(all.equal(unname(predictions), outer(x, expected), tolerance = 1e-6)))
  }
  invisible(NULL)
}

test_rewards_only_preparation()
test_one_predictor_elastic_net_objective()
cat("All block residual modeling tests passed.\n")
