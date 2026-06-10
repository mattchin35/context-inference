# Regression tests for R entry-script source path handling.

get_test_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    stop("Could not determine test script path from commandArgs().")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

# Read a script and verify that preprocessing files are loaded through the
# repository path resolver rather than getwd()-relative source() calls.
#
# Args:
#   script_path: character(1). Absolute or relative path to an R entry script.
#     Units: filesystem path.
#
# Returns:
#   TRUE invisibly if the script has no direct source("preprocessing/...") calls.
assert_no_direct_preprocessing_source <- function(script_path) {
  script_lines <- readLines(script_path, warn = FALSE)
  direct_source_pattern <- "^[[:space:]]*source\\([\"']preprocessing/"
  direct_source_lines <- grep(direct_source_pattern, script_lines, value = TRUE)
  
  if (length(direct_source_lines) > 0) {
    stop(
      "Entry scripts must use source_rcode() for preprocessing files. ",
      "Direct source() calls found in ", script_path, ": ",
      paste(direct_source_lines, collapse = " | ")
    )
  }
  
  invisible(TRUE)
}

test_script_path <- get_test_script_path()
repo_root <- normalizePath(
  file.path(dirname(test_script_path), "..", "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
rcode_root <- file.path(repo_root, "src", "Rcode")

entry_scripts <- c(
  file.path(rcode_root, "inspect_trials.R"),
  file.path(rcode_root, "session_behavior_analysis.R"),
  file.path(rcode_root, "session_behavior_analysis_mini.R"),
  file.path(rcode_root, "multisession_block_analysis.R")
)

for (entry_script in entry_scripts) {
  assert_no_direct_preprocessing_source(entry_script)
}

cat("All entry-script source path tests passed.\n")
