# Scientific contracts for the two per-session SAC TTS elastic-net fits.

# Return this script's absolute path (character scalar; filesystem units).
get_tts_modeling_test_path <- function() {
  arguments <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  stopifnot(length(arguments) == 1L)
  normalizePath(sub("^--file=", "", arguments), mustWork = TRUE)
}

# Return 81 CSV-like block rows from one session; trial counts are nonnegative.
# Rewards span 1:10 on each side; left has a higher intercept and reward slope.
make_tts_modeling_fixture <- function() {
  rewards <- c(0, rep(rep(seq_len(10), each = 2), 4))
  left <- c(FALSE, rep(c(FALSE, TRUE), 40))
  response <- round(2 + 0.7 * rewards + left * (2 + 0.8 * rewards) +
                      rep(c(-1, 0, 1), length.out = length(rewards)))
  data.frame(
    mouse = "CT024", session_id = "CT024_synthetic",
    block_ix = seq_along(rewards) - 1L,
    block_type = ifelse(left, "left_uncued", "right_uncued"),
    trials_to_correct = response, prev_n_rewarded = rewards,
    stringsAsFactors = FALSE
  )
}

# Assert an error whose message matches a character pattern; returns NULL.
expect_tts_modeling_error <- function(expression, pattern) {
  error <- tryCatch(eval(substitute(expression), parent.frame()), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), ignore.case = TRUE))
  invisible(NULL)
}

# Check summary coordinates, exact row alignment and fitted scientific signals.
# Input: no arguments. Return: invisible NULL on success.
test_tts_session_fit_contract <- function() {
  input <- make_tts_modeling_fixture()
  original <- input
  result <- fit_session_tts(input, seed = 123)
  stopifnot(identical(input, original))
  stopifnot(all(c("blocks", "audit", "parameters", "predictions", "cv", "status") %in% names(result)))
  stopifnot(nrow(result$blocks) == 80L, nrow(result$audit) == 81L)
  parameters <- result$parameters
  predictions <- result$predictions
  stopifnot(nrow(parameters) == 4L, nrow(predictions) == 320L)
  stopifnot(setequal(parameters$model_formula, c("rewards_only", "rewards_x_side")))
  stopifnot(setequal(parameters$lambda_choice, c("lambda.min", "lambda.1se")))
  stopifnot(all(c("mouse", "session", "model_formula", "lambda_choice", "lambda_value",
                  "intercept", "reward_slope", "residual_sd", "residual_rmse",
                  "coefficient_intercept", "coefficient_prev_n_rewarded",
                  "coefficient_block_side_left", "coefficient_prev_n_rewarded_block_side_left",
                  "left_intercept", "right_intercept", "left_reward_slope", "right_reward_slope") %in%
                  names(parameters)))
  numeric_columns <- c("intercept", "reward_slope", "residual_sd", "lambda_value")
  stopifnot(all(vapply(parameters[numeric_columns], is.numeric, logical(1))))
  stopifnot(all(c("row_id", "mouse", "session", "tts", "nrew", "side", "model_formula",
                  "lambda_choice", "predicted_tts", "residual") %in% names(predictions)))
  for (index in seq_len(nrow(parameters))) {
    parameter <- parameters[index, ]
    rows <- predictions[predictions$model_formula == parameter$model_formula &
                          predictions$lambda_choice == parameter$lambda_choice, ]
    stopifnot(identical(rows$row_id, result$blocks$row_id))
    expected <- parameter$coefficient_intercept + parameter$coefficient_prev_n_rewarded * rows$nrew +
      (rows$side == "left") * (parameter$coefficient_block_side_left +
                                 parameter$coefficient_prev_n_rewarded_block_side_left * rows$nrew)
    stopifnot(isTRUE(all.equal(rows$predicted_tts, as.numeric(expected), tolerance = 1e-7)))
    stopifnot(isTRUE(all.equal(rows$residual, rows$tts - rows$predicted_tts)))
    stopifnot(isTRUE(all.equal(parameter$residual_sd, stats::sd(rows$residual))))
    stopifnot(isTRUE(all.equal(parameter$residual_rmse, sqrt(mean(rows$residual^2)))))
  }
  summary <- parameters[parameters$model_formula == "rewards_only" &
                          parameters$lambda_choice == "lambda.min", ]
  stopifnot(summary$reward_slope > 0.7, summary$reward_slope < 1.4)
  stopifnot(summary$coefficient_block_side_left == 0,
            summary$coefficient_prev_n_rewarded_block_side_left == 0)
  stopifnot(summary$intercept == summary$coefficient_intercept,
            summary$reward_slope == summary$coefficient_prev_n_rewarded)
  interaction <- parameters[parameters$model_formula == "rewards_x_side" &
                              parameters$lambda_choice == "lambda.min", ]
  stopifnot(interaction$left_reward_slope > interaction$right_reward_slope + 0.3)
  stopifnot(interaction$left_intercept > interaction$right_intercept)
  stopifnot(interaction$residual_sd < summary$residual_sd)
  stopifnot(all(result$status$status == "ok"), nrow(result$status) == 2L)
  invisible(NULL)
}

# Check seeded shared folds, usable CV curves, and no leaked RNG mutation.
# Input: no arguments. Return: invisible NULL on success.
test_tts_session_cv_reproducibility <- function() {
  set.seed(987)
  initial_state <- .Random.seed
  first <- fit_session_tts(make_tts_modeling_fixture(), seed = 111)
  stopifnot(identical(.Random.seed, initial_state))
  second <- fit_session_tts(make_tts_modeling_fixture(), seed = 111)
  stopifnot(identical(first$parameters, second$parameters),
            identical(first$predictions, second$predictions), identical(first$cv, second$cv))
  folds <- first$cv$folds
  stopifnot(is.data.frame(folds), identical(folds$row_id, first$blocks$row_id))
  stopifnot(all(c("row_id", "fold_id") %in% names(folds)), length(unique(folds$fold_id)) >= 3L)
  stopifnot(is.data.frame(first$cv$curves))
  stopifnot(all(c("model_formula", "lambda", "cvm", "cvsd") %in% names(first$cv$curves)))
  stopifnot(setequal(first$cv$curves$model_formula, c("rewards_only", "rewards_x_side")))
  stopifnot(all(is.finite(first$cv$curves$cvm)), all(first$cv$curves$lambda > 0))
  different <- fit_session_tts(make_tts_modeling_fixture(), seed = 222)
  stopifnot(!identical(first$cv$folds$fold_id, different$cv$folds$fold_id))
  prepared <- prepare_tts_blocks(make_tts_modeling_fixture())$blocks
  repeated_preparation <- fit_session_tts(prepared, seed = 111)
  stopifnot(identical(first$blocks$row_id, repeated_preparation$blocks$row_id))
  stopifnot(identical(first$parameters, repeated_preparation$parameters))
  invisible(NULL)
}

# Check real-unit fitting on rescaled reward counts without side effects.
# Input: no arguments. Return: invisible NULL on success.
test_tts_session_original_units <- function() {
  original <- make_tts_modeling_fixture()
  transformed <- original
  transformed$prev_n_rewarded <- 10 * transformed$prev_n_rewarded + 20
  first <- fit_session_tts(original, seed = 123)$parameters
  second <- fit_session_tts(transformed, seed = 123)$parameters
  keep <- first$model_formula == "rewards_only"
  stopifnot(isTRUE(all.equal(second$reward_slope[keep], first$reward_slope[keep] / 10,
                             tolerance = 1e-6)))
  stopifnot(isTRUE(all.equal(second$intercept[keep],
                            first$intercept[keep] - 2 * first$reward_slope[keep], tolerance = 1e-6)))
  stopifnot(isTRUE(all.equal(second$residual_sd[keep], first$residual_sd[keep], tolerance = 1e-6)))
  invisible(NULL)
}

# Check failures stay visible and a missing side does not suppress side-free fits.
# Input: no arguments. Return: invisible NULL on success.
test_tts_session_statuses <- function() {
  small <- make_tts_modeling_fixture()[1:4, ]
  result <- fit_session_tts(small, seed = 123)
  stopifnot(all(result$status$status == "insufficient_blocks"))
  stopifnot(nrow(result$parameters) == 4L, all(is.na(result$parameters$intercept)))
  stopifnot(is.numeric(result$parameters$intercept), nrow(result$predictions) == 0L)
  constant <- make_tts_modeling_fixture()
  constant$trials_to_correct <- 2
  stopifnot(all(fit_session_tts(constant, seed = 123)$status$status == "constant_response"))
  constant <- make_tts_modeling_fixture()
  constant$prev_n_rewarded <- 2
  stopifnot(all(fit_session_tts(constant, seed = 123)$status$status == "constant_rewards"))
  single_side <- make_tts_modeling_fixture()
  single_side$block_type <- "right_uncued"
  result <- fit_session_tts(single_side, seed = 123)
  stopifnot(result$status$status[result$status$model_formula == "rewards_x_side"] == "missing_side")
  stopifnot(result$status$status[result$status$model_formula == "rewards_only"] == "ok")
  stopifnot(all(is.na(result$parameters$left_reward_slope[
    result$parameters$model_formula == "rewards_x_side"])))
  constant_side <- make_tts_modeling_fixture()
  constant_side$prev_n_rewarded[constant_side$block_type == "left_uncued"] <- 2
  result <- fit_session_tts(constant_side, seed = 123)
  stopifnot(result$status$status[result$status$model_formula == "rewards_x_side"] ==
              "insufficient_side_variation")
  empty <- fit_session_tts(make_tts_modeling_fixture()[FALSE, ], seed = 123)
  stopifnot(nrow(empty$parameters) == 4L, nrow(empty$predictions) == 0L)
  invalid <- make_tts_modeling_fixture()
  invalid$trials_to_correct[3] <- -1
  invalid$prev_n_rewarded[4] <- Inf
  result <- fit_session_tts(invalid, seed = 123)
  stopifnot(nrow(result$blocks) == 78L, !any(result$blocks$block_ix %in% c(0, 2, 3)))
  invisible(NULL)
}

# Reject calls that accidentally pool sessions or omit the required random seed.
# Input: no arguments. Return: invisible NULL on success.
test_tts_session_invalid_calls <- function() {
  input <- make_tts_modeling_fixture()
  expect_tts_modeling_error(fit_session_tts(input), "seed")
  expect_tts_modeling_error(fit_session_tts(input, seed = 123, alpha = 2), "alpha")
  input$session_id[2] <- "CT024_other"
  expect_tts_modeling_error(fit_session_tts(input, seed = 123), "session")
  invisible(NULL)
}

rcode_root <- dirname(dirname(get_tts_modeling_test_path()))
source(file.path(rcode_root, "tts_data.R"))
implementation <- file.path(rcode_root, "tts_modeling.R")
if (file.exists(implementation)) source(implementation)
if (!exists("fit_session_tts", mode = "function")) {
  stop("fit_session_tts() has not been implemented.")
}
stopifnot(requireNamespace("glmnet", quietly = TRUE))
test_tts_session_fit_contract()
test_tts_session_cv_reproducibility()
test_tts_session_original_units()
test_tts_session_statuses()
test_tts_session_invalid_calls()
cat("All TTS session modeling tests passed.\n")
