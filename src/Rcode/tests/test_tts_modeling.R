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

# Generate 656 block rows in 16 sessions across four mice with a recorded seed.
# Args: seed integer(1). Returns: data.frame(656,p); TTS/rewards in trial units.
make_tts_population_fixture <- function(seed = 392L) {
  had_rng <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_rng) saved_rng <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_rng) assign(".Random.seed", saved_rng, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  })
  set.seed(seed)
  rows <- list()
  for (mouse_index in seq_len(4)) {
    for (session_index in seq_len(4)) {
      reward <- c(0, rep(rep(seq_len(10), each = 2), 2))
      left <- c(FALSE, rep(c(FALSE, TRUE), 20))
      deviations <- stats::rnorm(4, sd = c(1, 0.12, 0.8, 0.15))
      tts <- 12 + mouse_index / 2 + deviations[1] +
        (0.5 + deviations[2]) * reward +
        left * (1 + deviations[3] + deviations[4] * reward) +
        stats::rnorm(length(reward), sd = 0.8)
      rows[[length(rows) + 1L]] <- data.frame(
        mouse = paste0("SYN", mouse_index),
        session_id = paste0("SYN", mouse_index, "_session", session_index),
        block_ix = seq_along(reward) - 1L,
        block_type = ifelse(left, "left_uncued", "right_uncued"),
        trials_to_correct = pmax(round(tts), 0), prev_n_rewarded = reward,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

# Verify exact candidates, common block identities, ML and numeric fit summaries.
# Args: result list returned by fit_population_tts; returns invisible NULL.
test_tts_population_contract <- function(result) {
  stopifnot(all(c("models", "comparison", "blocks", "audit", "random_effects", "meta") %in%
                  names(result)))
  model_ids <- paste0("M", seq_len(4))
  stopifnot(identical(names(result$models), model_ids))
  stopifnot(nrow(result$blocks) == 640L, nrow(result$audit) == 656L)
  comparison <- result$comparison
  stopifnot(identical(comparison$model_id, model_ids))
  stopifnot(all(c("formula", "nobs", "n_sessions", "n_mice", "n_parameters",
                  "logLik", "AIC", "BIC", "delta_AIC", "delta_BIC", "converged",
                  "singular", "status", "message", "likelihood_method") %in% names(comparison)))
  stopifnot(all(comparison$nobs == 640L), all(comparison$n_sessions == 16L),
            all(comparison$n_mice == 4L), all(comparison$likelihood_method == "ML"))
  expected_formulas <- c(
    "tts ~ nrew * side + (nrew * side | session)",
    "tts ~ nrew + side + (nrew * side | session)",
    "tts ~ nrew + side + (nrew + side | session)",
    "tts ~ nrew + side + (nrew * side | mouse/session)"
  )
  for (index in seq_along(model_ids)) {
    model <- result$models[[index]]
    stopifnot(inherits(model, "merMod"), !lme4::isREML(model))
    actual_formula <- paste(deparse(stats::formula(model)), collapse = " ")
    stopifnot(identical(gsub("[[:space:]]", "", actual_formula),
                        gsub("[[:space:]]", "", expected_formulas[index])))
    stopifnot(identical(rownames(stats::model.frame(model)), result$blocks$row_id))
    stopifnot(isTRUE(all.equal(comparison$AIC[index], stats::AIC(model))))
    stopifnot(isTRUE(all.equal(comparison$BIC[index], stats::BIC(model))))
    stopifnot(isTRUE(all.equal(comparison$logLik[index], as.numeric(stats::logLik(model)))))
    stopifnot(comparison$singular[index] == lme4::isSingular(model))
    if (comparison$singular[index]) {
      stopifnot(grepl("singular", paste(comparison$status[index], comparison$message[index]),
                     ignore.case = TRUE))
    }
  }
  stopifnot(isTRUE(all.equal(comparison$delta_AIC, comparison$AIC - min(comparison$AIC))))
  stopifnot(isTRUE(all.equal(comparison$delta_BIC, comparison$BIC - min(comparison$BIC))))
  stopifnot(is.logical(comparison$converged), is.logical(comparison$singular))
  stopifnot(all(c("model_id", "grp", "var1", "var2", "vcov", "sdcor") %in%
                  names(result$random_effects)))
  stopifnot(all(c("generator", "analysis_version", "units") %in% names(result$meta)))
  invisible(NULL)
}

# Compare population/session prediction types against lme4 with stable row keys.
# Args: result list returned by fit_population_tts; returns invisible NULL.
test_tts_population_predictions <- function(result) {
  input <- result$blocks[c(1, 7, 41, 182), ]
  predictions <- predict_population_tts(result, input)
  stopifnot(nrow(predictions) == nrow(input) * 8L)
  stopifnot(all(c("row_id", "model_id", "prediction_type", "predicted_tts", "status", "message") %in%
                  names(predictions)))
  stopifnot(setequal(predictions$prediction_type, c("population", "conditional")))
  for (model_id in names(result$models)) {
    for (prediction_type in c("population", "conditional")) {
      selected <- predictions[predictions$model_id == model_id &
                                predictions$prediction_type == prediction_type, ]
      direct <- stats::predict(result$models[[model_id]], newdata = input,
                                re.form = if (prediction_type == "population") NA else NULL)
      stopifnot(identical(selected$row_id, input$row_id), all(selected$status == "ok"))
      stopifnot(isTRUE(all.equal(selected$predicted_tts, unname(direct), tolerance = 1e-10)))
    }
  }
  unknown <- input[1, , drop = FALSE]
  unknown$session <- "unknown_session"
  unknown$row_id <- "unknown_row"
  predictions <- predict_population_tts(result, unknown)
  conditional <- predictions[predictions$prediction_type == "conditional", ]
  population <- predictions[predictions$prediction_type == "population", ]
  stopifnot(all(conditional$status == "unknown_group"), all(is.na(conditional$predicted_tts)))
  stopifnot(all(population$status == "ok"), all(is.finite(population$predicted_tts)))
  invisible(NULL)
}

# Check failed candidates remain visible, with no synthetic fallback or Inf deltas.
# Input: no arguments. Return: invisible NULL.
test_tts_population_fit_failures <- function() {
  input <- make_tts_modeling_fixture()
  result <- fit_population_tts(input)
  stopifnot(length(result$models) == 4L, all(vapply(result$models, is.null, logical(1))))
  stopifnot(all(result$comparison$status == "fit_error"))
  stopifnot(all(nzchar(result$comparison$message)), all(is.na(result$comparison$AIC)))
  stopifnot(all(is.na(result$comparison$delta_AIC)), all(is.na(result$comparison$delta_BIC)))
  predictions <- predict_population_tts(result, result$blocks[1:2, ])
  stopifnot(all(predictions$status == "fit_error"), all(is.na(predictions$predicted_tts)))
  invisible(NULL)
}

# Verify fixed exemplar errors separately from fitted models, preserving trial units.
# Input: no arguments. Return: invisible NULL.
test_tts_exemplar_predictions <- function() {
  blocks <- prepare_tts_blocks(make_tts_modeling_fixture())$blocks[1:3, ]
  exemplars <- data.frame(name = c("flat", "slope"), intercept = c(2, 4),
                          reward_slope = c(0, 1), residual_sd = c(2, 5))
  result <- predict_tts_exemplars(blocks, exemplars)
  stopifnot(nrow(result) == 6L)
  stopifnot(all(c("row_id", "exemplar", "tts", "nrew", "predicted_tts", "residual",
                  "normalized_residual") %in% names(result)))
  for (index in seq_len(nrow(exemplars))) {
    selected <- result[result$exemplar == exemplars$name[index], ]
    expected <- exemplars$intercept[index] + exemplars$reward_slope[index] * blocks$nrew
    stopifnot(identical(selected$row_id, blocks$row_id))
    stopifnot(isTRUE(all.equal(selected$predicted_tts, expected)))
    stopifnot(isTRUE(all.equal(selected$residual, blocks$tts - expected)))
    stopifnot(isTRUE(all.equal(selected$normalized_residual,
                              (blocks$tts - expected) / exemplars$residual_sd[index])))
  }
  invalid <- exemplars
  invalid$residual_sd[1] <- 0
  expect_tts_modeling_error(predict_tts_exemplars(blocks, invalid), "residual_sd")
  invalid <- exemplars
  invalid$name[2] <- invalid$name[1]
  expect_tts_modeling_error(predict_tts_exemplars(blocks, invalid), "name")
  invisible(NULL)
}

if (!exists("fit_population_tts", mode = "function")) {
  stop("fit_population_tts() has not been implemented.")
}
stopifnot(requireNamespace("lme4", quietly = TRUE))
population_fixture <- make_tts_population_fixture()
population_result <- fit_population_tts(population_fixture)
test_tts_population_contract(population_result)
test_tts_population_predictions(population_result)
test_tts_population_fit_failures()
test_tts_exemplar_predictions()
cat("All TTS population and exemplar tests passed.\n")
