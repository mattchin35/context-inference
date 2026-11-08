# Numeric-source and saved-result rendering tests for SAC TTS figures.

# Return this script's absolute path (character scalar, filesystem units).
get_tts_plotting_test_path <- function() {
  arguments <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  stopifnot(length(arguments) == 1L)
  normalizePath(sub("^--file=", "", arguments), mustWork = TRUE)
}

# Return one 61-row synthetic session; counts are trials/rewarded trials.
# Args: mouse/session character(1), index integer(1) controlling true effects.
make_tts_plotting_session <- function(mouse, session, index) {
  block <- 0:60
  reward <- c(0, rep(rep(1:10, each = 2), 3))
  left <- rep(c(FALSE, TRUE), length.out = length(block))
  tts <- round(1 + index / 2 + (0.15 + index / 12) * reward +
                 left * (index / 3 + reward * index / 30) +
                 rep(c(-1, 0, 1, 0, 1, -1), length.out = length(block)))
  data.frame(mouse = mouse, session_id = session, block_ix = block,
             block_type = ifelse(left, "left_uncued", "right_uncued"),
             trials_to_correct = pmax(0, tts), prev_n_rewarded = reward,
             stringsAsFactors = FALSE)
}

# Save synthetic single-session and multi-mouse artifacts for rendering tests.
# Args: root character(1), existing owned directory under ignored project data/.
# Returns: list(single,cohort,result), two saved directory paths and cohort list.
make_tts_plotting_artifacts <- function(root) {
  sessions <- list()
  for (index in seq_len(6)) {
    mouse <- paste0("CT", c("016", "024", "025")[(index + 1L) %/% 2L])
    session <- paste0(mouse, "_synthetic_", index)
    result <- fit_session_tts(make_tts_plotting_session(mouse, session, index), seed = 100 + index)
    result$exemplars <- default_tts_exemplars()
    result$exemplar_predictions <- predict_tts_exemplars(result$blocks, result$exemplars)
    result$population <- NULL
    sessions[[session]] <- result
  }
  paths <- data.frame(mouse = vapply(sessions, tts_plot_fixture_mouse, ""),
                      session = names(sessions), result_dir = "synthetic_only", stringsAsFactors = FALSE)
  cohort <- combine_tts_sessions(sessions, paths)
  cohort$meta$seed <- 100
  cohort$meta$alpha <- 0.5
  cohort$meta$min_valid_blocks <- 5L
  list(single = save_tts_results(sessions[[1]], file.path(root, "single")),
       cohort = save_tts_results(cohort, file.path(root, "cohort")), result = cohort)
}

# Return the scalar mouse label from a synthetic session result (no units).
tts_plot_fixture_mouse <- function(result) result$audit$mouse[1]

# Assert an expression errors with a matching pattern; returns invisible NULL.
expect_tts_plotting_error <- function(expression, pattern) {
  error <- tryCatch(eval(substitute(expression), parent.frame()), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), ignore.case = TRUE))
  invisible(NULL)
}

# Check the source data use side-free primary coefficients and exact exemplar refs.
# Args: result saved cohort contract. Returns invisible NULL on success.
test_tts_plot_parameter_sources <- function(result) {
  sources <- tts_parameter_distribution_data(result$parameters, result$exemplars)
  stopifnot(all(c("source", "parameter", "value", "name", "mouse", "session") %in% names(sources)))
  stopifnot(setequal(sources$parameter, c("intercept", "reward_slope", "residual_sd")))
  primary <- result$parameters[result$parameters$model_formula == "rewards_only" &
                                 result$parameters$lambda_choice == "lambda.min", ]
  for (parameter in c("intercept", "reward_slope", "residual_sd")) {
    rows <- sources[sources$source == "session" & sources$parameter == parameter, ]
    stopifnot(nrow(rows) == nrow(primary), !anyDuplicated(rows$session))
    stopifnot(isTRUE(all.equal(rows$value, primary[[parameter]][match(rows$session, primary$session)])))
    references <- sources[sources$source == "exemplar" & sources$parameter == parameter, ]
    stopifnot(nrow(references) == 4L)
    stopifnot(isTRUE(all.equal(references$value,
                              result$exemplars[[parameter]][match(references$name, result$exemplars$name)])))
  }
  comparison <- plot_tts_model_comparison(result$population$comparison)
  stopifnot(inherits(comparison, "ggplot"))
  stopifnot(all(c("criterion", "delta", "model_id", "diagnostic") %in% names(comparison$data)))
  for (criterion in c("AIC", "BIC")) {
    rows <- comparison$data[comparison$data$criterion == criterion, ]
    stopifnot(isTRUE(all.equal(rows$delta,
      result$population$comparison[[paste0("delta_", criterion)]][match(rows$model_id,
        result$population$comparison$model_id)])))
  }
  one_session <- result$blocks$session[1]
  blocks <- result$blocks[result$blocks$session == one_session, ]
  predictions <- result$predictions[result$predictions$session == one_session, ]
  plot <- plot_tts_session_fit(blocks, predictions)
  stopifnot(inherits(plot, "ggplot"))
  stopifnot(grepl("rewarded", plot$labels$x, ignore.case = TRUE),
            grepl("trials", plot$labels$y, ignore.case = TRUE))
  invisible(NULL)
}

# Verify mean error and sample SD remain separate scientific quantities.
# Input: none. Returns invisible NULL; residuals/reference SD are in trials.
test_tts_plot_exemplar_errors <- function() {
  predictions <- data.frame(mouse = "CT024", session = "example",
                             exemplar = rep(c("a", "b"), each = 2),
                             residual = c(-2, 2, 4, 4), exemplar_residual_sd = 2)
  summary <- tts_exemplar_error_data(predictions)
  stopifnot(all(c("mouse", "session", "exemplar", "mean_error", "residual_sd", "reference_sd") %in% names(summary)))
  stopifnot(summary$mean_error[summary$exemplar == "a"] == 0,
            summary$mean_error[summary$exemplar == "b"] == 4)
  stopifnot(isTRUE(all.equal(summary$residual_sd[summary$exemplar == "a"], sqrt(8))),
            summary$residual_sd[summary$exemplar == "b"] == 0,
            all(summary$reference_sd == 2))
  invisible(NULL)
}

# Render every figure group from saved results while all fit calls are disabled.
# Args: root owned directory, artifacts from make_tts_plotting_artifacts().
# Returns manifest invisibly; PNG/source/caption paths must be explicit/nonempty.
test_tts_plot_saved_rendering <- function(root, artifacts) {
  original_session_fit <- fit_session_tts
  original_population_fit <- fit_population_tts
  assign("fit_session_tts", tts_forbidden_fit, envir = .GlobalEnv)
  assign("fit_population_tts", tts_forbidden_fit, envir = .GlobalEnv)
  on.exit({
    assign("fit_session_tts", original_session_fit, envir = .GlobalEnv)
    assign("fit_population_tts", original_population_fit, envir = .GlobalEnv)
  })
  input_files <- list.files(artifacts$cohort, recursive = TRUE, full.names = TRUE)
  before <- tools::md5sum(input_files)
  output <- file.path(root, "cohort_figures")
  manifest <- plot_tts_results(artifacts$cohort, output_dir = output)
  stopifnot(is.data.frame(manifest))
  stopifnot(all(c("figure_id", "path", "source_path", "caption_path", "status", "message") %in% names(manifest)))
  rendered <- manifest[manifest$status == "ok", ]
  stopifnot(setequal(sub("_.*$", "", rendered$figure_id), paste0("F", 1:9)))
  stopifnot(all(file.exists(rendered$path)), all(file.info(rendered$path)$size > 1000))
  stopifnot(all(grepl("[.]png$", rendered$path)), all(startsWith(rendered$path, "/")))
  stopifnot(all(file.exists(rendered$source_path)), all(file.exists(rendered$caption_path)))
  captions <- vapply(rendered$caption_path, tts_read_caption, "")
  stopifnot(all(nchar(captions) > 20), all(grepl("session|mouse|mice", captions, ignore.case = TRUE)))
  stopifnot(all(file.exists(file.path(output, c("figure_manifest.csv", "render_metadata.rds",
                                               "scripts/tts_plotting.R", "scripts/plot_tts_results.R")))))
  saved_sources <- utils::read.csv(rendered$source_path[grep("^F5", rendered$figure_id)[1]])
  stopifnot(setequal(saved_sources$source, c("session", "exemplar")))
  stopifnot(identical(before, tools::md5sum(input_files)))
  prior_hash <- tools::md5sum(rendered$path)
  expect_tts_plotting_error(plot_tts_results(artifacts$cohort, output_dir = output), "exist|overwrite")
  stopifnot(identical(prior_hash, tools::md5sum(rendered$path)))
  invisible(manifest)
}

# A fit sentinel proves plotting does not invoke computation routines.
# Args: any R inputs. Returns: never; always raises an error.
tts_forbidden_fit <- function(...) stop("Plotting must not refit models.")

# Return a sidecar caption's contents as one character scalar, no physical units.
tts_read_caption <- function(path) paste(readLines(path), collapse = " ")

# Check single-session skips and fresh default render directories preserve files.
# Args: artifacts saved fixture contract. Returns invisible NULL.
test_tts_plot_single_and_skipped <- function(artifacts) {
  first <- plot_tts_results(artifacts$single, figures = c("F2", "F3", "F4", "F5"))
  stopifnot(all(first$status[sub("_.*$", "", first$figure_id) %in% c("F2", "F3")] == "skipped"))
  stopifnot(all(grepl("population|single", first$message[first$status == "skipped"], ignore.case = TRUE)))
  ok <- first[first$status == "ok", ]
  stopifnot(any(grepl("^F4", ok$figure_id)), any(grepl("^F5", ok$figure_id)))
  f5 <- first[grepl("^F5", first$figure_id), ]
  stopifnot(grepl("single|one session|1 session", tts_read_caption(f5$caption_path[1]), ignore.case = TRUE))
  before <- tools::md5sum(ok$path)
  second <- plot_tts_results(artifacts$single, figures = "F4")
  stopifnot(!any(second$path[second$status == "ok"] %in% ok$path))
  stopifnot(identical(before, tools::md5sum(ok$path)))
  invisible(NULL)
}

# Check sessions without eligible fitted parameters produce informative skips.
# Args: root owned scratch directory. Returns invisible NULL on success.
test_tts_plot_empty_fit <- function(root) {
  result <- fit_session_tts(make_tts_plotting_session("CT024", "empty_fit", 1)[1:3, ], seed = 100)
  result$population <- NULL
  result$exemplars <- default_tts_exemplars()
  result$exemplar_predictions <- predict_tts_exemplars(result$blocks, result$exemplars)
  saved <- save_tts_results(result, file.path(root, "empty_fit"))
  manifest <- plot_tts_results(saved, output_dir = file.path(root, "empty_figures"))
  stopifnot(is.data.frame(manifest), all(manifest$status %in% c("ok", "skipped")))
  stopifnot(all(manifest$status[grepl("^F5", manifest$figure_id)] == "skipped"))
  stopifnot(all(nzchar(manifest$message[manifest$status == "skipped"])))
  invisible(NULL)
}

rcode_root <- dirname(dirname(get_tts_plotting_test_path()))
source(file.path(rcode_root, "run_tts_batch.R"))
for (file in c("tts_plotting.R", "plot_tts_results.R")) {
  candidate <- file.path(rcode_root, file)
  if (file.exists(candidate)) source(candidate)
}
if (!exists("plot_tts_results", mode = "function")) stop("plot_tts_results() has not been implemented.")
stopifnot(requireNamespace("ggplot2", quietly = TRUE))

test_root <- tempfile("w5_tests_", tmpdir = file.path(dirname(dirname(rcode_root)), "data"))
dir.create(test_root, recursive = TRUE)
tryCatch({
  artifacts <- make_tts_plotting_artifacts(test_root)
  test_tts_plot_parameter_sources(artifacts$result)
  test_tts_plot_exemplar_errors()
  test_tts_plot_saved_rendering(test_root, artifacts)
  test_tts_plot_single_and_skipped(artifacts)
  test_tts_plot_empty_fit(test_root)
  cat("All TTS plotting tests passed.\n")
}, finally = unlink(test_root, recursive = TRUE))
