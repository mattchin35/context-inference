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

fit_result <- add_block_residual_model_outputs(block_df, seed = 123, min_valid_blocks = 5)
stopifnot(all(c(
  "lasso_lambda_min_residual_TTS",
  "lasso_lambda_1se_residual_TTS",
  "elastic_net_lambda_min_residual_TTS",
  "elastic_net_lambda_1se_residual_TTS"
) %in% names(fit_result$block_df)))
stopifnot(nrow(fit_result$summary_df) == 4)
stopifnot(all(c(
  "coefficient_intercept",
  "coefficient_prev_n_rewarded",
  "coefficient_block_side_left",
  "coefficient_prev_n_rewarded_block_side_left"
) %in% names(fit_result$summary_df)))

small_result <- add_block_residual_model_outputs(block_df[1:4, ], seed = 123, min_valid_blocks = 5)
stopifnot(all(small_result$block_df$lasso_lambda_min_residual_TTS == "None"))
stopifnot(all(small_result$summary_df$coefficient_intercept == "None"))
stopifnot(identical(small_result$summary_df$n_valid_blocks, c(4L, 4L, 4L, 4L)))

cat("All block residual modeling tests passed.\n")
