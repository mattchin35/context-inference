# Result persistence and computation-entry integration tests for SAC TTS.

# Return this script's absolute path, a character scalar in filesystem units.
get_tts_results_test_path <- function() {
  arguments <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  stopifnot(length(arguments) == 1L)
  normalizePath(sub("^--file=", "", arguments), mustWork = TRUE)
}

# Make one deterministic 25-block raw session table; counts are in trials.
# Args: mouse/session character scalars, offset numeric scalar.
# Returns: data.frame(25,6), including session-initial block zero.
make_tts_results_session <- function(mouse = "CT024", session = "CT024_test_1", offset = 0) {
  block <- 0:24
  reward <- c(0, rep(c(1, 3, 6, 9, 12, 15), 4))
  side <- rep(c("right_uncued", "left_uncued"), length.out = length(block))
  tts <- round(2 + offset + 0.4 * reward + 2 * (side == "left_uncued") +
                 rep(c(-1, 0, 1, 0, 1), length.out = length(block)))
  data.frame(mouse = mouse, session_id = session, block_ix = block,
             block_type = side, trials_to_correct = tts, prev_n_rewarded = reward,
             stringsAsFactors = FALSE)
}

# Assert an unevaluated expression errors with a matching message; no units.
# Args: R expression, character(1) pattern. Returns invisible NULL.
expect_tts_results_error <- function(expression, pattern) {
  error <- tryCatch(eval(substitute(expression), parent.frame()), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), ignore.case = TRUE))
  invisible(NULL)
}

# Check exact RDS round trip, numeric exports, script snapshots and no overwrite.
# Args: root character(1) scratch directory within ignored project data/.
# Returns invisible NULL; removes nothing outside this test's scratch tree.
test_tts_result_round_trip <- function(root) {
  result <- fit_session_tts(make_tts_results_session(), seed = 97)
  result$population <- NULL
  result$exemplars <- default_tts_exemplars()
  result$exemplar_predictions <- predict_tts_exemplars(result$blocks, result$exemplars)
  directory <- file.path(root, "roundtrip")
  saved <- save_tts_results(result, directory)
  stopifnot(identical(normalizePath(saved), normalizePath(directory)))
  stopifnot(identical(load_tts_results(saved), result))
  files <- c("result.rds", "blocks.csv", "audit.csv", "parameters.csv", "predictions.csv",
             "status.csv", "cv_folds.csv", "cv_curves.csv", "exemplars.csv",
             "exemplar_predictions.csv", "run.log", "SUMMARY.md",
             "scripts/run_tts_session.R", "scripts/run_tts_batch.R")
  stopifnot(all(file.exists(file.path(saved, files))))
  parameter_export <- utils::read.csv(file.path(saved, "parameters.csv"))
  stopifnot(isTRUE(all.equal(parameter_export$intercept, result$parameters$intercept)))
  before <- tools::md5sum(file.path(saved, files))
  expect_tts_results_error(save_tts_results(result, saved), "exist|overwrite")
  stopifnot(identical(before, tools::md5sum(file.path(saved, files))))
  invisible(NULL)
}

# Check code/input/fitting/version changes all appear in plain cache metadata.
# Args: root character(1) owned scratch tree. Returns invisible NULL.
test_tts_cache_metadata <- function(root) {
  input <- file.path(root, "cache_input.csv")
  code <- file.path(root, "cache_source.R")
  utils::write.csv(make_tts_results_session(), input, row.names = FALSE)
  writeLines("source_version <- 1", code)
  first <- tts_cache_metadata(input, "test_v1", 97, 0.5, 5L, source_files = code)
  same <- tts_cache_metadata(input, "test_v1", 97, 0.5, 5L, source_files = code)
  stopifnot(identical(first, same))
  stopifnot(identical(unname(first$input_checksums), unname(tools::md5sum(input))))
  stopifnot(identical(unname(first$source_checksums), unname(tools::md5sum(code))))
  stopifnot(!identical(first, tts_cache_metadata(input, "test_v2", 97, 0.5, 5L, source_files = code)))
  stopifnot(!identical(first, tts_cache_metadata(input, "test_v1", 98, 0.5, 5L, source_files = code)))
  stopifnot(!identical(first, tts_cache_metadata(input, "test_v1", 97, 1, 5L, source_files = code)))
  stopifnot(!identical(first, tts_cache_metadata(input, "test_v1", 97, 0.5, 6L, source_files = code)))
  writeLines("source_version <- 2", code)
  stopifnot(!identical(first, tts_cache_metadata(input, "test_v1", 97, 0.5, 5L, source_files = code)))
  writeLines("source_version <- 1", code)
  table <- make_tts_results_session()
  table$trials_to_correct[2] <- table$trials_to_correct[2] + 1
  utils::write.csv(table, input, row.names = FALSE)
  stopifnot(!identical(first, tts_cache_metadata(input, "test_v1", 97, 0.5, 5L, source_files = code)))
  invisible(NULL)
}

# Check single-session cache reuse, reruns, source preservation and provenance.
# Args: root character(1) owned scratch tree. Returns invisible NULL.
test_tts_session_runner <- function(root) {
  input <- file.path(root, "runner_input.csv")
  output <- file.path(root, "session_runs")
  utils::write.csv(make_tts_results_session(), input, row.names = FALSE)
  input_hash <- tools::md5sum(input)
  planned <- run_tts_session(input, output, seed = 97, dry_run = TRUE)
  stopifnot(is.list(planned), identical(planned$input_csv, normalizePath(input)))
  stopifnot(!dir.exists(output), identical(input_hash, tools::md5sum(input)))
  first <- run_tts_session(input, output, seed = 97)
  first_hash <- tools::md5sum(file.path(first, "result.rds"))
  cached <- run_tts_session(input, output, seed = 97)
  stopifnot(identical(first, cached), identical(first_hash, tools::md5sum(file.path(first, "result.rds"))))
  repeated <- run_tts_session(input, output, seed = 97, rerun = TRUE)
  versioned <- run_tts_session(input, output, seed = 97, analysis_version = "test_v2")
  penalized <- run_tts_session(input, output, seed = 97, alpha = 1)
  stopifnot(length(unique(c(first, repeated, versioned, penalized))) == 4L)
  stopifnot(identical(input_hash, tools::md5sum(input)))
  stopifnot(identical(first_hash, tools::md5sum(file.path(first, "result.rds"))))
  result <- load_tts_results(first)
  stopifnot(is.null(result$population), nrow(result$exemplars) == 4L)
  stopifnot(identical(result$exemplars$name, c("expert_inf", "expert_rl", "naive_inf", "naive_rl")))
  stopifnot(identical(result$exemplars$intercept, c(2, 2, 6, 4)))
  stopifnot(identical(result$exemplars$reward_slope, c(0, 0.5, 0, 1)))
  stopifnot(identical(result$exemplars$residual_sd, c(2, 2, 5, 5)))
  stopifnot(nrow(result$exemplar_predictions) == 4L * nrow(result$blocks))
  stopifnot(is.list(result$meta$cache_key), length(result$meta$cache_key$source_checksums) >= 4L)
  stopifnot(length(result$meta$package_versions) >= 2L, !is.null(result$meta$r_version))
  log <- paste(readLines(file.path(first, "run.log")), collapse = "\n")
  summary <- paste(readLines(file.path(first, "SUMMARY.md")), collapse = "\n")
  stopifnot(grepl("elapsed", log, ignore.case = TRUE), grepl("97", log))
  stopifnot(grepl("CT024_test_1", log), grepl("CT024_test_1", summary))
  stopifnot(grepl("run_tts_session.R", summary, fixed = TRUE))
  invisible(NULL)
}

# Check discovery is explicit, ordered, latent-only and optionally date-filtered.
# Args: root character(1) owned scratch tree. Returns paths data.frame(n,3).
test_tts_discovery <- function(root) {
  data_root <- file.path(root, "discovery")
  directories <- c("CT024/CT024_20260609_latent_inference/processed",
                   "CT024/CT024_20260611_latent_inference/processed",
                   "CT016/CT016_20260520_latent_inference/processed",
                   "CT024/CT024_20260612_other_task/processed",
                   "CT099/CT099_20260609_latent_inference/processed")
  mice <- c("CT024", "CT024", "CT016", "CT024", "CT099")
  sessions <- c("CT024_2026-06-09_143852", "CT024_2026-06-11_142023", "CT016_2026-05-20_100000",
                "CT024_2026-06-12_100000", "CT099_2026-06-09_100000")
  for (index in seq_along(directories)) {
    directory <- file.path(data_root, directories[index])
    dir.create(directory, recursive = TRUE)
    utils::write.csv(make_tts_results_session(mice[index], sessions[index]),
                     file.path(directory, paste0(sessions[index], "_block_performance.csv")), row.names = FALSE)
  }
  # Actual processed folders also contain HMM-annotated derivatives of this table.
  primary <- file.path(data_root, directories[1], paste0(sessions[1], "_block_performance.csv"))
  for (derivative in c("dynamax_exemplar_hmm", "exemplar_likelihood_hmm")) {
    file.copy(primary, file.path(dirname(primary), paste0(sessions[1], "_", derivative, "_block_performance.csv")))
  }
  paths <- discover_tts_sessions(data_root)
  stopifnot(is.data.frame(paths), nrow(paths) == 3L)
  stopifnot(all(c("mouse", "session", "input_csv") %in% names(paths)))
  stopifnot(!anyDuplicated(paths$session), identical(paths$mouse, c("CT016", "CT024", "CT024")))
  filtered <- discover_tts_sessions(data_root, mice = "CT024", dates = "2026-06-09")
  stopifnot(nrow(filtered) == 1L, filtered$session == sessions[1])
  duplicate <- file.path(dirname(paths$input_csv[1]), "duplicate_block_performance.csv")
  file.copy(paths$input_csv[1], duplicate)
  expect_tts_results_error(discover_tts_sessions(data_root), "ambig|multiple|duplicate")
  unlink(duplicate)
  paths
}

# Check reproducible serial/parallel batch tables and saved population outputs.
# Args: root character(1) owned scratch tree. Returns invisible NULL.
test_tts_batch_runner <- function(root) {
  source_root <- file.path(root, "batch_inputs")
  dir.create(source_root)
  paths <- data.frame(mouse = rep(c("CT016", "CT024"), each = 2),
                      session = c("CT016_a", "CT016_b", "CT024_a", "CT024_b"),
                      input_csv = file.path(source_root, paste0("session_", 1:4, ".csv")))
  for (index in seq_len(nrow(paths))) {
    utils::write.csv(make_tts_results_session(paths$mouse[index], paths$session[index], index / 2),
                     paths$input_csv[index], row.names = FALSE)
  }
  before <- tools::md5sum(paths$input_csv)
  planned_root <- file.path(root, "batch_planned")
  planned <- run_tts_batch(paths, planned_root, seed = 73, workers = 2L, dry_run = TRUE)
  stopifnot(is.list(planned), nrow(planned$session_paths) == 4L, !dir.exists(planned_root))
  serial <- run_tts_batch(paths, file.path(root, "batch_serial"), seed = 73, workers = 1L)
  parallel <- run_tts_batch(paths, file.path(root, "batch_parallel"), seed = 73, workers = 2L, rerun = TRUE)
  left <- load_tts_results(serial)
  right <- load_tts_results(parallel)
  stopifnot(isTRUE(all.equal(left$parameters, right$parameters, check.attributes = FALSE)))
  stopifnot(isTRUE(all.equal(left$predictions, right$predictions, check.attributes = FALSE)))
  stopifnot(identical(left$cv$folds, right$cv$folds))
  stopifnot(all(c("session", "mouse", "row_id", "fold_id") %in% names(left$cv$folds)))
  stopifnot(all(c("session", "mouse", "model_formula", "lambda", "cvm", "cvsd") %in% names(left$cv$curves)))
  stopifnot(nrow(left$population$comparison) == 4L, nrow(left$population$predictions) == 8L * nrow(left$blocks))
  stopifnot(nrow(left$exemplar_predictions) == 4L * nrow(left$blocks))
  stopifnot(length(left$sessions) == 4L, nrow(left$session_paths) == 4L)
  stopifnot(all(c("session", "mouse", "result_dir") %in% names(left$session_paths)))
  stopifnot(all(dir.exists(left$session_paths$result_dir)))
  stopifnot(all(grepl("sac_tts", left$session_paths$result_dir, fixed = TRUE)))
  exports <- c("population_comparison.csv", "population_predictions.csv", "population_random_effects.csv", "session_paths.csv")
  stopifnot(all(file.exists(file.path(serial, exports))))
  stopifnot(identical(before, tools::md5sum(paths$input_csv)))
  invisible(NULL)
}

# Check failed scientific runs retain error logs without overwriting the input.
# Args: root character(1) owned scratch tree. Returns invisible NULL.
test_tts_failed_run_log <- function(root) {
  input <- file.path(root, "invalid_identity.csv")
  bad <- make_tts_results_session()
  bad$mouse <- NULL
  utils::write.csv(bad, input, row.names = FALSE)
  before <- tools::md5sum(input)
  output <- file.path(root, "failed_runs")
  expect_tts_results_error(run_tts_session(input, output), "mouse")
  logs <- list.files(output, pattern = "run[.]log$", full.names = TRUE, recursive = TRUE)
  stopifnot(length(logs) == 1L)
  stopifnot(any(grepl("error", readLines(logs), ignore.case = TRUE)))
  stopifnot(identical(before, tools::md5sum(input)))
  invisible(NULL)
}

rcode_root <- dirname(dirname(get_tts_results_test_path()))
source(file.path(rcode_root, "tts_modeling.R"))
for (filename in c("tts_results.R", "run_tts_session.R", "run_tts_batch.R")) {
  candidate <- file.path(rcode_root, filename)
  if (file.exists(candidate)) source(candidate)
}
required_functions <- c("save_tts_results", "load_tts_results", "tts_cache_metadata", "default_tts_exemplars",
                        "run_tts_session", "run_tts_batch", "discover_tts_sessions")
for (function_name in required_functions) {
  if (!exists(function_name, mode = "function")) stop(function_name, "() has not been implemented.")
}

# Scratch artifacts stay inside ignored project data/, then are removed.
test_root <- tempfile("w4_tests_", tmpdir = file.path(dirname(dirname(rcode_root)), "data"))
dir.create(test_root, recursive = TRUE)
tryCatch({
  test_tts_result_round_trip(test_root)
  test_tts_cache_metadata(test_root)
  test_tts_session_runner(test_root)
  test_tts_discovery(test_root)
  test_tts_batch_runner(test_root)
  test_tts_failed_run_log(test_root)
  cat("All TTS result and runner tests passed.\n")
}, finally = unlink(test_root, recursive = TRUE))
