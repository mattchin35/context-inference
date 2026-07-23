# Regularized block-level residual models for trials-to-correct analysis.

BLOCK_RESIDUAL_MODEL_SPECS <- list(
  list(model_type = "lasso", alpha = 1.0),
  list(model_type = "elastic_net", alpha = 0.5)
)
BLOCK_RESIDUAL_FORMULA_SPECS <- list(
  list(
    model_formula = "rewards_x_side",
    formula = stats::as.formula("trials_to_correct ~ prev_n_rewarded * block_side"),
    write_legacy_aliases = TRUE
  ),
  list(
    model_formula = "rewards_plus_side",
    formula = stats::as.formula("trials_to_correct ~ prev_n_rewarded + block_side"),
    write_legacy_aliases = FALSE
  )
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

# Prepare a glmnet matrix for a configured block residual formula.
#
# Args:
#   block_df: data.frame with shape (n_blocks, n_columns). Required columns are
#     `trials_to_correct`, `prev_n_rewarded`, and `block_type`.
#   model_formula: formula identifier. Supported values are "rewards_x_side"
#     and "rewards_plus_side".
#
# Returns:
#   list with fields `x`, `y`, `valid_rows`, and `model_df`. `x` has shape
#   (n_valid_blocks, n_features), `y` has length n_valid_blocks, and
#   `valid_rows` uses 1-based row positions from `block_df`.
prepare_block_residual_model_dataframe <- function(block_df, model_formula = "rewards_x_side") {
  required_columns <- c("trials_to_correct", "prev_n_rewarded", "block_type")
  formula_spec <- NULL
  for (spec in BLOCK_RESIDUAL_FORMULA_SPECS) {
    if (identical(spec$model_formula, model_formula)) {
      formula_spec <- spec
    }
  }
  if (is.null(formula_spec)) {
    stop(paste("Unknown block residual model formula:", model_formula))
  }
  missing_columns <- setdiff(required_columns, names(block_df))
  if (length(missing_columns) > 0) {
    return(list(
      x = matrix(numeric(0), nrow = 0, ncol = 0),
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
    formula_spec$formula,
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
#   data.frame with one row and raw MAD, R-style scaled MAD, IQR, RMSE, and
#   sample SD.
compute_residual_spread_metrics <- function(residuals) {
  residual_values <- residuals[is.finite(residuals)]
  if (length(residual_values) == 0) {
    return(data.frame(
      residual_mad_raw = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_mad_scaled = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_iqr = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_rmse = BLOCK_RESIDUAL_MISSING_VALUE,
      residual_sd = BLOCK_RESIDUAL_MISSING_VALUE,
      stringsAsFactors = FALSE
    ))
  }
  residual_median <- stats::median(residual_values)
  mad_raw <- stats::median(abs(residual_values - residual_median))
  residual_sd <- if (length(residual_values) >= 2) {
    as.numeric(stats::sd(residual_values))
  } else {
    BLOCK_RESIDUAL_MISSING_VALUE
  }
  data.frame(
    residual_mad_raw = as.numeric(mad_raw),
    residual_mad_scaled = as.numeric(mad_raw * BLOCK_RESIDUAL_SCALED_MAD_CONSTANT),
    residual_iqr = as.numeric(stats::IQR(residual_values)),
    residual_rmse = as.numeric(sqrt(mean(residual_values^2))),
    residual_sd = residual_sd,
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

block_residual_formula_prefix <- function(model_formula, model_type, lambda_choice) {
  lambda_tag <- ifelse(lambda_choice == "lambda.min", "lambda_min", "lambda_1se")
  paste(model_type, model_formula, lambda_tag, sep = "_")
}

block_residual_formula_column <- function(model_formula, model_type, lambda_choice) {
  paste0(block_residual_formula_prefix(model_formula, model_type, lambda_choice), "_residual_TTS")
}

derive_side_specific_coefficients <- function(coefficient_values) {
  intercept <- coefficient_values$coefficient_intercept
  reward_slope <- coefficient_values$coefficient_prev_n_rewarded
  left_offset <- coefficient_values$coefficient_block_side_left
  left_reward_delta <- coefficient_values$coefficient_prev_n_rewarded_block_side_left
  data.frame(
    right_intercept = as.numeric(intercept),
    left_intercept = as.numeric(intercept + left_offset),
    right_reward_slope = as.numeric(reward_slope),
    left_reward_slope = as.numeric(reward_slope + left_reward_delta),
    side_intercept_delta_left_minus_right = as.numeric(left_offset),
    side_reward_slope_delta_left_minus_right = as.numeric(left_reward_delta),
    stringsAsFactors = FALSE
  )
}

missing_summary_row <- function(model_formula, model_type, alpha, lambda_choice, n_valid_blocks) {
  metrics <- compute_residual_spread_metrics(numeric(0))
  data.frame(
    model_formula = model_formula,
    model_type = model_type,
    lambda_choice = lambda_choice,
    lambda_value = BLOCK_RESIDUAL_MISSING_VALUE,
    alpha = alpha,
    n_valid_blocks = as.integer(n_valid_blocks),
    coefficient_intercept = BLOCK_RESIDUAL_MISSING_VALUE,
    coefficient_prev_n_rewarded = BLOCK_RESIDUAL_MISSING_VALUE,
    coefficient_block_side_left = BLOCK_RESIDUAL_MISSING_VALUE,
    coefficient_prev_n_rewarded_block_side_left = BLOCK_RESIDUAL_MISSING_VALUE,
    right_intercept = BLOCK_RESIDUAL_MISSING_VALUE,
    left_intercept = BLOCK_RESIDUAL_MISSING_VALUE,
    right_reward_slope = BLOCK_RESIDUAL_MISSING_VALUE,
    left_reward_slope = BLOCK_RESIDUAL_MISSING_VALUE,
    side_intercept_delta_left_minus_right = BLOCK_RESIDUAL_MISSING_VALUE,
    side_reward_slope_delta_left_minus_right = BLOCK_RESIDUAL_MISSING_VALUE,
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

fit_one_block_glmnet_model <- function(prepared, model_formula, model_type, alpha, lambda_choice, seed, nfolds) {
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
  coefficient_values <- data.frame(
    coefficient_intercept = extract_coefficient(coefficient_matrix, "(Intercept)"),
    coefficient_prev_n_rewarded = extract_coefficient(coefficient_matrix, "prev_n_rewarded"),
    coefficient_block_side_left = extract_coefficient(coefficient_matrix, "block_sideleft"),
    coefficient_prev_n_rewarded_block_side_left = extract_coefficient(
      coefficient_matrix,
      "prev_n_rewarded:block_sideleft"
    ),
    stringsAsFactors = FALSE
  )
  side_specific_coefficients <- derive_side_specific_coefficients(coefficient_values)
  summary_df <- data.frame(
    model_formula = model_formula,
    model_type = model_type,
    lambda_choice = lambda_choice,
    lambda_value = if (lambda_choice == "lambda.min") cv_fit$lambda.min else cv_fit$lambda.1se,
    alpha = alpha,
    n_valid_blocks = as.integer(length(prepared$y)),
    coefficient_values,
    side_specific_coefficients,
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
  for (formula_spec in BLOCK_RESIDUAL_FORMULA_SPECS) {
    for (spec in BLOCK_RESIDUAL_MODEL_SPECS) {
      for (lambda_choice in BLOCK_RESIDUAL_LAMBDA_CHOICES) {
        output_df[[block_residual_formula_column(
          formula_spec$model_formula,
          spec$model_type,
          lambda_choice
        )]] <- BLOCK_RESIDUAL_MISSING_VALUE
        if (isTRUE(formula_spec$write_legacy_aliases)) {
          output_df[[block_residual_column(spec$model_type, lambda_choice)]] <- BLOCK_RESIDUAL_MISSING_VALUE
        }
      }
    }
  }
  summary_rows <- list()
  for (formula_spec in BLOCK_RESIDUAL_FORMULA_SPECS) {
    prepared <- prepare_block_residual_model_dataframe(
      output_df,
      model_formula = formula_spec$model_formula
    )
    n_valid_blocks <- length(prepared$y)
    if (n_valid_blocks < min_valid_blocks) {
      for (spec in BLOCK_RESIDUAL_MODEL_SPECS) {
        for (lambda_choice in BLOCK_RESIDUAL_LAMBDA_CHOICES) {
          summary_rows[[length(summary_rows) + 1]] <- missing_summary_row(
            formula_spec$model_formula,
            spec$model_type,
            spec$alpha,
            lambda_choice,
            n_valid_blocks
          )
        }
      }
      next
    }
    nfolds <- min(10, n_valid_blocks)
    for (spec in BLOCK_RESIDUAL_MODEL_SPECS) {
      for (lambda_choice in BLOCK_RESIDUAL_LAMBDA_CHOICES) {
        fit_result <- fit_one_block_glmnet_model(
          prepared,
          model_formula = formula_spec$model_formula,
          model_type = spec$model_type,
          alpha = spec$alpha,
          lambda_choice = lambda_choice,
          seed = seed,
          nfolds = nfolds
        )
        output_df[prepared$valid_rows, block_residual_formula_column(
          formula_spec$model_formula,
          spec$model_type,
          lambda_choice
        )] <- fit_result$residuals
        if (isTRUE(formula_spec$write_legacy_aliases)) {
          output_df[prepared$valid_rows, block_residual_column(spec$model_type, lambda_choice)] <- fit_result$residuals
        }
        summary_rows[[length(summary_rows) + 1]] <- fit_result$summary_df
      }
    }
  }
  list(block_df = output_df, summary_df = do.call(rbind, summary_rows))
}

# Save combined and formula-specific residual-model summary CSVs for one session.
#
# Args:
#   summary_df: data.frame with one row per model/lambda pair and a
#     `model_formula` column.
#   session_save_path: directory where the session's processed outputs are saved.
#   sess_id: full session identifier used in output file names.
#
# Returns:
#   named character vector of saved CSV paths. The `combined` entry preserves
#   the original filename convention.
save_block_residual_model_summaries <- function(summary_df, session_save_path, sess_id) {
  combined_path <- save_block_residual_model_summary(summary_df, session_save_path, sess_id)
  saved_paths <- c(combined = combined_path)
  if (!"model_formula" %in% names(summary_df)) {
    return(saved_paths)
  }
  for (formula_spec in BLOCK_RESIDUAL_FORMULA_SPECS) {
    formula_rows <- summary_df[summary_df$model_formula == formula_spec$model_formula, , drop = FALSE]
    formula_path <- file.path(
      session_save_path,
      paste0(sess_id, "_block_residual_model_summary_", formula_spec$model_formula, ".csv")
    )
    utils::write.csv(
      formula_rows,
      formula_path,
      row.names = FALSE,
      na = BLOCK_RESIDUAL_MISSING_VALUE
    )
    saved_paths[[formula_spec$model_formula]] <- formula_path
  }
  saved_paths
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
