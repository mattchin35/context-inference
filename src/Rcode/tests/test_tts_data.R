# Tests for block eligibility and row identity in SAC TTS analyses.

# Return this test script's absolute filesystem path, a character scalar.
get_tts_data_test_path <- function() {
  arguments <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  stopifnot(length(arguments) == 1L)
  normalizePath(sub("^--file=", "", arguments), mustWork = TRUE)
}

# Return a 10-row CSV-like block table; counts are trials/rewarded trials.
# Rows deliberately mix sessions and omit prev_n_correct, which is not required.
make_tts_data_fixture <- function() {
  data.frame(
    mouse = c(rep("CT024", 8), rep("CT016", 2)),
    session_id = c(rep("CT024_2026-06-09_143852", 8), rep("CT016_example", 2)),
    block_ix = as.character(c(2, 0, 1, 3, 4, 5, 6, 7, 12, 10)),
    block_type = c("right_uncued", "right_uncued", "left_uncued", "left_uncued",
                   "left_uncued", "unknown", "right_uncued", "left_uncued",
                   "left_uncued", "right_uncued"),
    trials_to_correct = c("0", "None", "2", "Inf", "3", "1", "-1", "1", "4", "2"),
    prev_n_rewarded = c("7", "0", "5", "6", "None", "2", "3", "-1", "11", "0"),
    untouched_note = paste0("original_", seq_len(10)),
    stringsAsFactors = FALSE
  )
}

# Evaluate expression and assert an error whose message contains pattern.
# Inputs: unevaluated R expression and character scalar; returns invisible NULL.
expect_tts_data_error <- function(expression, pattern) {
  error <- tryCatch(eval(substitute(expression), parent.frame()), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), ignore.case = TRUE))
  invisible(NULL)
}

# Check the retained rows, zero convention, units and all-row audit contract.
# Input: no arguments. Return: invisible NULL on success.
test_tts_eligibility_and_alignment <- function() {
  input <- make_tts_data_fixture()
  original <- input
  result <- prepare_tts_blocks(input)
  stopifnot(identical(input, original), is.list(result))
  stopifnot(all(c("blocks", "audit") %in% names(result)))
  blocks <- result$blocks
  audit <- result$audit
  stopifnot(is.data.frame(blocks), is.data.frame(audit), nrow(audit) == nrow(input))
  stopifnot(all(c("mouse", "session", "block_ix", "row_id", "tts", "nrew",
                  "side", "is_first_block") %in% names(blocks)))
  stopifnot(all(c("mouse", "session", "block_ix", "row_id", "included",
                  "exclusion_reason", "is_first_block") %in% names(audit)))
  stopifnot(identical(blocks$untouched_note, input$untouched_note[c(1, 3, 9)]))
  stopifnot(is.numeric(blocks$tts), identical(as.numeric(blocks$tts), c(0, 2, 4)))
  stopifnot(is.numeric(blocks$nrew), identical(as.numeric(blocks$nrew), c(7, 5, 11)))
  stopifnot(is.numeric(blocks$block_ix), identical(as.numeric(blocks$block_ix), c(2, 1, 12)))
  stopifnot(is.factor(blocks$side), identical(levels(blocks$side), c("right", "left")))
  stopifnot(identical(as.character(blocks$side), c("right", "left", "left")))
  stopifnot(identical(audit$included, seq_len(nrow(input)) %in% c(1, 3, 9)))
  stopifnot(identical(audit$is_first_block, seq_len(nrow(input)) %in% c(2, 10)))
  stopifnot(is.character(audit$row_id), !anyNA(audit$row_id), !anyDuplicated(audit$row_id))
  stopifnot(identical(blocks$row_id, audit$row_id[audit$included]))
  stopifnot(all(audit$exclusion_reason[audit$included] == ""))
  stopifnot(grepl("first_block", audit$exclusion_reason[2]))
  stopifnot(grepl("invalid_tts", audit$exclusion_reason[2]))
  stopifnot(grepl("invalid_tts", audit$exclusion_reason[4]))
  stopifnot(grepl("invalid_nrew", audit$exclusion_reason[5]))
  stopifnot(grepl("unknown_side", audit$exclusion_reason[6]))
  stopifnot(grepl("invalid_tts", audit$exclusion_reason[7]))
  stopifnot(grepl("invalid_nrew", audit$exclusion_reason[8]))
  stopifnot(grepl("first_block", audit$exclusion_reason[10]))
  invisible(NULL)
}

# Check first-block opt-out, stable keys and absent optional previous-correct data.
# Input: no arguments. Return: invisible NULL on success.
test_tts_first_block_and_stable_keys <- function() {
  input <- make_tts_data_fixture()
  retained_first <- prepare_tts_blocks(input, exclude_first_block = FALSE)
  stopifnot(identical(retained_first$blocks$untouched_note, input$untouched_note[c(1, 3, 9, 10)]))
  stopifnot(retained_first$blocks$is_first_block[4])
  stopifnot(!retained_first$audit$included[2])
  expected <- prepare_tts_blocks(input)$audit
  shuffled <- prepare_tts_blocks(input[10:1, ])$audit
  stopifnot(identical(expected$row_id, rev(shuffled$row_id)))
  input$prev_n_correct <- "None"
  stopifnot(identical(prepare_tts_blocks(input)$blocks$row_id, expected$row_id[expected$included]))
  invisible(NULL)
}

# Check actual legacy ID coexistence and alternative full-ID/mouse columns.
# Input: no arguments. Return: invisible NULL on success.
test_tts_identity_aliases <- function() {
  input <- make_tts_data_fixture()[1:3, ]
  input$session_ID <- "CT024_2026-06-09"
  prepared <- prepare_tts_blocks(input)
  stopifnot(all(prepared$blocks$session == "CT024_2026-06-09_143852"))
  stopifnot(all(prepared$blocks$session_ID == "CT024_2026-06-09"))
  input$source_session_id <- input$session_id
  input$source_mouse <- input$mouse
  input$session_id <- NULL
  input$mouse <- NULL
  aliased <- prepare_tts_blocks(input)
  stopifnot(identical(prepared$blocks$row_id, aliased$blocks$row_id))
  input$source_session_id <- NULL
  legacy <- prepare_tts_blocks(input)
  stopifnot(all(legacy$blocks$session == "CT024_2026-06-09"))
  invisible(NULL)
}

# Check concatenated-table source indices restore original within-session keys.
# Input: no arguments. Return: invisible NULL on success.
test_tts_source_block_indices <- function() {
  input <- make_tts_data_fixture()[1:3, ]
  input$source_block_ix <- input$block_ix
  input$block_ix <- as.numeric(input$block_ix) + 100
  result <- prepare_tts_blocks(input)
  stopifnot(identical(as.numeric(result$blocks$block_ix), c(2, 1)))
  stopifnot(identical(as.numeric(result$blocks$input_block_ix), c(102, 101)))
  stopifnot(identical(as.numeric(result$blocks$source_block_ix), c(2, 1)))
  stopifnot(identical(result$audit$is_first_block, c(FALSE, TRUE, FALSE)))
  invisible(NULL)
}

# Check malformed identity is rejected rather than silently dropping/merging rows.
# Input: no arguments. Return: invisible NULL on success.
test_tts_invalid_identity <- function() {
  input <- make_tts_data_fixture()
  duplicate <- rbind(input, input[1, ])
  expect_tts_data_error(prepare_tts_blocks(duplicate), "duplicate")
  input$mouse[1] <- ""
  expect_tts_data_error(prepare_tts_blocks(input), "mouse")
  input <- make_tts_data_fixture()
  input$session_id[1] <- "None"
  expect_tts_data_error(prepare_tts_blocks(input), "session")
  input <- make_tts_data_fixture()
  input$block_ix[1] <- "not_a_block"
  expect_tts_data_error(prepare_tts_blocks(input), "block")
  input <- make_tts_data_fixture()
  input$mouse <- NULL
  expect_tts_data_error(prepare_tts_blocks(input), "mouse")
  input <- make_tts_data_fixture()
  input$source_session_id <- input$session_id
  input$source_session_id[1] <- "CT024_another_session"
  expect_tts_data_error(prepare_tts_blocks(input), "conflict")
  invisible(NULL)
}

# Check empty/all-excluded data keep named, typed outputs rather than erroring.
# Input: no arguments. Return: invisible NULL on success.
test_tts_empty_outputs <- function() {
  input <- make_tts_data_fixture()
  empty <- prepare_tts_blocks(input[FALSE, ])
  stopifnot(nrow(empty$blocks) == 0L, nrow(empty$audit) == 0L)
  stopifnot(is.numeric(empty$blocks$tts), is.numeric(empty$blocks$nrew))
  stopifnot(is.logical(empty$audit$included), is.character(empty$audit$exclusion_reason))
  excluded <- prepare_tts_blocks(input[2, , drop = FALSE])
  stopifnot(nrow(excluded$blocks) == 0L, nrow(excluded$audit) == 1L)
  invisible(NULL)
}

rcode_root <- dirname(dirname(get_tts_data_test_path()))
implementation <- file.path(rcode_root, "tts_data.R")
if (file.exists(implementation)) source(implementation)
if (!exists("prepare_tts_blocks", mode = "function")) {
  stop("prepare_tts_blocks() has not been implemented.")
}

test_tts_eligibility_and_alignment()
test_tts_first_block_and_stable_keys()
test_tts_identity_aliases()
test_tts_source_block_indices()
test_tts_invalid_identity()
test_tts_empty_outputs()
cat("All TTS data tests passed.\n")
