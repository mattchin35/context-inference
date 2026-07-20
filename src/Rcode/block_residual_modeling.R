# Regularized block-level residual models for trials-to-correct analysis.

BLOCK_RESIDUAL_MODEL_SPECS <- list(
  list(model_type = "lasso", alpha = 1.0),
  list(model_type = "elastic_net", alpha = 0.5)
)
BLOCK_RESIDUAL_LAMBDA_CHOICES <- c("lambda.min", "lambda.1se")
BLOCK_RESIDUAL_MISSING_VALUE <- "None"
BLOCK_RESIDUAL_SCALED_MAD_CONSTANT <- 1.4826

# Map block type labels to a right-reference side factor.
#
# Args:
#   block_type_values: vector with length n_blocks. Values beginning with
#     "right" map to "right" and values beginning with "left" map to "left".
#
# Returns:
#   factor with length n_blocks and levels c("right", "left"). Unknown labels
#   are NA. The level order makes right the model reference level.
derive_block_side <- function(block_type_values) {
  normalized <- tolower(trimws(as.character(block_type_values)))
  block_side <- rep(NA_character_, length(normalized))
  block_side[startsWith(normalized, "right")] <- "right"
  block_side[startsWith(normalized, "left")] <- "left"
  factor(block_side, levels = c("right", "left"))
}

# Prepare a glmnet matrix for trials_to_correct ~ prev_n_rewarded * block_side.
#
# Args:
#   block_df: data.frame with shape (n_blocks, n_columns). Required columns are
#     `trials_to_correct`, `prev_n_rewarded`, and `block_type`.
#
# Returns:
#   list with fields `x`, `y`, `valid_rows`, and `model_df`. `x` has shape
#   (n_valid_blocks, 3), `y` has length n_valid_blocks, and `valid_rows` uses
#   1-based row positions from `block_df`.
prepare_block_residual_model_dataframe <- function(block_df) {
  required_columns <- c("trials_to_correct", "prev_n_rewarded", "block_type")
  missing_columns <- setdiff(required_columns, names(block_df))
  if (length(missing_columns) > 0) {
    return(list(
      x = matrix(numeric(0), nrow = 0, ncol = 3),
      y = numeric(0),
      valid_rows = integer(0),
      model_df = data.frame()
    ))
  }
  model_df <- data.frame(
    trials_to_correct = suppressWarnings(as.numeric(block_df$trials_to_correct)),
    prev_n_rewarded = suppressWarnings(as.numeric(block_df$prev_n_rewarded)),
    block_side = derive_block_side(block_df$block_type)
  )
  valid_rows <- which(stats::complete.cases(model_df))
  if (length(valid_rows) == 0) {
    return(list(
      x = matrix(numeric(0), nrow = 0, ncol = 3),
      y = numeric(0),
      valid_rows = integer(0),
      model_df = model_df[valid_rows, , drop = FALSE]
    ))
  }
  valid_model_df <- model_df[valid_rows, , drop = FALSE]
  x <- stats::model.matrix(
    trials_to_correct ~ prev_n_rewarded * block_side,
    data = valid_model_df
  )[, -1, drop = FALSE]
  y <- valid_model_df$trials_to_correct
  list(x = x, y = y, valid_rows = valid_rows, model_df = valid_model_df)
}

# Compute residual spread metrics in trial units.
#
# Args:
#   residuals: numeric vector with length n_valid_blocks. Residuals are observed
#     trials-to-correct minus fitted trials-to-correct.
#
# Returns:
#   data.frame with one row and raw MAD, R-style scaled MAD, IQR, and RMSE.
compute_residual_spread_metrics <- function(residuals) {
  residual_values <- residuals[is.finite(residuals)]
  if (length(residual_values) == 0) {
    return(data.frame(
      residual_mad_raw = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_mad_scaled = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_iqr = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_rmse = BLOCK_RESIDUAL_MISSING_VALUE,
      stringsAsFactors = FALSE
    ))
  }
  residual_median <- stats::median(residual_values)
  mad_raw <- stats::median(abs(residual_values - residual_median))
  data.frame(
    residual_mad_raw = as.numeric(mad_raw),
    residual_mad_scaled = as.numeric(mad_raw * BLOCK_RESIDUAL_SCALED_MAD_CONSTANT),
    residual_iqr = as.numeric(stats::IQR(residual_values)),
    residual_rmse = as.numeric(sqrt(mean(residual_values^2))),
    stringsAsFactors = FALSE
  )
}

block_residual_prefix <- function(model_type, lambda_choice) {
  lambda_tag <- ifelse(lambda_choice == "lambda.min", "lambda_min", "lambda_1se")
  paste(model_type, lambda_tag, sep = "_")
}

block_residual_column <- function(model_type, lambda_choice) {
  paste0(block_residual_prefix(model_type, lambda_choice), "_residual_TTS")
}

missing_summary_row <- function(model_type, alpha, lambda_choice, n_valid_blocks) {
  metrics <- compute_residual_spread_metrics(numeric(0))
  data.frame(
    model_type = model_type,
    lambda_choice = lambda_choice,
    lambda_value = BLOCK_RESIDUAL_MISSING_VALUE,
    alpha = alpha,
    n_valid_blocks = as.integer(n_valid_blocks),
    coefficient_intercept = BLOCK_RESIDUAL_MISSING_VALUE,
    coefficient_prev_n_rewarded = BLOCK_RESIDUAL_MISSING_VALUE,
    coefficient_block_side_left = BLOCK_RESIDUAL_MISSING_VALUE,
    coefficient_prev_n_rewarded_block_side_left = BLOCK_RESIDUAL_MISSING_VALUE,
    metrics,
    stringsAsFactors = FALSE
  )
}

extract_coefficient <- function(coefficient_matrix, term_name) {
  if (!term_name %in% rownames(coefficient_matrix)) {
    return(0)
  }
  as.numeric(coefficient_matrix[term_name, 1])
}

fit_one_block_glmnet_model <- function(prepared, model_type, alpha, lambda_choice, seed, nfolds) {
  set.seed(seed)
  cv_fit <- glmnet::cv.glmnet(
    x = prepared$x,
    y = prepared$y,
    family = "gaussian",
    alpha = alpha,
    standardize = TRUE,
    nfolds = nfolds
  )
  predictions <- as.numeric(stats::predict(cv_fit, newx = prepared$x, s = lambda_choice))
  residuals <- prepared$y - predictions
  coefficient_matrix <- as.matrix(stats::coef(cv_fit, s = lambda_choice))
  metrics <- compute_residual_spread_metrics(residuals)
  summary_df <- data.frame(
    model_type = model_type,
    lambda_choice = lambda_choice,
    lambda_value = if (lambda_choice == "lambda.min") cv_fit$lambda.min else cv_fit$lambda.1se,
    alpha = alpha,
    n_valid_blocks = as.integer(length(prepared$y)),
    coefficient_intercept = extract_coefficient(coefficient_matrix, "(Intercept)"),
    coefficient_prev_n_rewarded = extract_coefficient(coefficient_matrix, "prev_n_rewarded"),
    coefficient_block_side_left = extract_coefficient(coefficient_matrix, "block_sideleft"),
    coefficient_prev_n_rewarded_block_side_left = extract_coefficient(
      coefficient_matrix,
      "prev_n_rewarded:block_sideleft"
    ),
    metrics,
    stringsAsFactors = FALSE
  )
  list(residuals = residuals, summary_df = summary_df)
}

# Add lasso and elastic-net residuals plus a combined model summary.
#
# Args:
#   block_df: data.frame with shape (n_blocks, n_columns). Required fitting
#     columns are `trials_to_correct`, `prev_n_rewarded`, and `block_type`.
#   seed: integer random seed for glmnet cross-validation folds.
#   min_valid_blocks: integer minimum number of valid rows needed for fitting.
#
# Returns:
#   list with `block_df` and `summary_df`. Residual columns use the project
#   missing sentinel "None" on rows that were not included in the model fit.
add_block_residual_model_outputs <- function(block_df, seed = 12345, min_valid_blocks = 5) {
  output_df <- block_df
  for (spec in BLOCK_RESIDUAL_MODEL_SPECS) {
    for (lambda_choice in BLOCK_RESIDUAL_LAMBDA_CHOICES) {
      output_df[[block_residual_column(spec$model_type, lambda_choice)]] <- BLOCK_RESIDUAL_MISSING_VALUE
    }
  }
  prepared <- prepare_block_residual_model_dataframe(output_df)
  n_valid_blocks <- length(prepared$y)
  if (n_valid_blocks < min_valid_blocks) {
    summary_rows <- list()
    for (spec in BLOCK_RESIDUAL_MODEL_SPECS) {
      for (lambda_choice in BLOCK_RESIDUAL_LAMBDA_CHOICES) {
        summary_rows[[length(summary_rows) + 1]] <- missing_summary_row(
          spec$model_type,
          spec$alpha,
          lambda_choice,
          n_valid_blocks
        )
      }
    }
    return(list(block_df = output_df, summary_df = do.call(rbind, summary_rows)))
  }
  nfolds <- min(10, n_valid_blocks)
  summary_rows <- list()
  for (spec in BLOCK_RESIDUAL_MODEL_SPECS) {
    for (lambda_choice in BLOCK_RESIDUAL_LAMBDA_CHOICES) {
      fit_result <- fit_one_block_glmnet_model(
        prepared,
        model_type = spec$model_type,
        alpha = spec$alpha,
        lambda_choice = lambda_choice,
        seed = seed,
        nfolds = nfolds
      )
      output_df[prepared$valid_rows, block_residual_column(spec$model_type, lambda_choice)] <- fit_result$residuals
      summary_rows[[length(summary_rows) + 1]] <- fit_result$summary_df
    }
  }
  list(block_df = output_df, summary_df = do.call(rbind, summary_rows))
}

# Save the combined residual-model summary CSV for one session.
#
# Args:
#   summary_df: data.frame with one row per model/lambda pair.
#   session_save_path: directory where the session's processed outputs are saved.
#   sess_id: full session identifier used in the output file name.
#
# Returns:
#   character path to the saved CSV.
save_block_residual_model_summary <- function(summary_df, session_save_path, sess_id) {
  summary_path <- file.path(
    session_save_path,
    paste0(sess_id, "_block_residual_model_summary.csv")
  )
  utils::write.csv(summary_df, summary_path, row.names = FALSE, na = BLOCK_RESIDUAL_MISSING_VALUE)
  summary_path
}
