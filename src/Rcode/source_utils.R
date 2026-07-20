# Helper utilities for sourcing files within src/Rcode.

.source_utils_load_wd <- getwd()

# Infer the current script path from the source() call stack or command-line
# execution context.
#
# Returns:
#   character(1) or NULL. Absolute path to the current script when available.
infer_current_script_path <- function() {
  frame_indices <- rev(seq_len(sys.nframe()))
  
  for (frame_index in frame_indices) {
    frame_file <- sys.frame(frame_index)$ofile
    if (!is.null(frame_file)) {
      return(normalizePath(frame_file, winslash = "/", mustWork = TRUE))
    }
  }
  
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    file_path <- sub("^--file=", "", file_arg[[1]])
    candidate_paths <- c(
      file_path,
      file.path(.source_utils_load_wd, file_path)
    )
    existing_paths <- candidate_paths[file.exists(candidate_paths)]
    if (length(existing_paths) > 0) {
      return(normalizePath(existing_paths[[1]], winslash = "/", mustWork = TRUE))
    }
  }
  
  NULL
}

# Locate the src/Rcode directory by walking up the filesystem tree.
#
# Args:
#   start_path: character(1). Existing file or directory path used as the
#     starting point for the upward search. Units: filesystem path.
#
# Returns:
#   character(1) or NULL. Absolute path to the src/Rcode directory, or NULL if
#   no matching directory is found.
find_rcode_root <- function(start_path) {
  normalized_start <- normalizePath(start_path, winslash = "/", mustWork = TRUE)
  current_path <- if (dir.exists(normalized_start)) normalized_start else dirname(normalized_start)
  
  repeat {
    if (file.exists(file.path(current_path, "Rcode.Rproj"))) {
      return(current_path)
    }
    
    parent_path <- dirname(current_path)
    if (identical(parent_path, current_path)) {
      return(NULL)
    }
    current_path <- parent_path
  }
}

# Resolve a path relative to src/Rcode without relying on getwd() alone.
#
# Args:
#   relative_path: character(1). Path relative to src/Rcode. Units: filesystem
#     path.
#   caller_path: character(1) or NULL. Optional file or directory path used to
#     seed the src/Rcode root search. Units: filesystem path.
#
# Returns:
#   character(1). Absolute path to the requested file inside src/Rcode.
resolve_rcode_path <- function(relative_path, caller_path = NULL) {
  if (!is.character(relative_path) || length(relative_path) != 1) {
    stop("relative_path must be a character scalar.")
  }
  
  candidate_roots <- character(0)
  inferred_caller_path <- infer_current_script_path()
  
  if (!is.null(inferred_caller_path)) {
    candidate_roots <- c(candidate_roots, find_rcode_root(inferred_caller_path))
  }
  
  if (!is.null(caller_path)) {
    candidate_roots <- c(candidate_roots, find_rcode_root(caller_path))
  }
  
  candidate_roots <- c(candidate_roots, find_rcode_root(getwd()))
  
  repo_relative_root <- file.path(getwd(), "src", "Rcode")
  if (dir.exists(repo_relative_root) && file.exists(file.path(repo_relative_root, "Rcode.Rproj"))) {
    candidate_roots <- c(candidate_roots, normalizePath(repo_relative_root, winslash = "/", mustWork = TRUE))
  }
  
  candidate_roots <- unique(candidate_roots[!is.na(candidate_roots) & nzchar(candidate_roots)])
  
  for (root_path in candidate_roots) {
    candidate_path <- file.path(root_path, relative_path)
    if (file.exists(candidate_path)) {
      return(normalizePath(candidate_path, winslash = "/", mustWork = TRUE))
    }
  }
  
  stop(
    "Could not resolve '", relative_path, "' inside src/Rcode. Checked roots: ",
    paste(candidate_roots, collapse = ", ")
  )
}

# Source a file relative to src/Rcode.
#
# Args:
#   relative_path: character(1). Path relative to src/Rcode. Units: filesystem
#     path.
#   caller_path: character(1) or NULL. Optional file or directory path used to
#     seed the src/Rcode root search. Units: filesystem path.
#   local: environment, logical scalar, or list. Evaluation target passed to
#     source(). Default: the caller's environment.
#   ...: Additional arguments passed to source().
#
# Returns:
#   The return value from source(), typically a named list containing the
#   sourced expressions and value.
source_rcode <- function(relative_path, caller_path = NULL, local = parent.frame(), ...) {
  resolved_path <- resolve_rcode_path(relative_path, caller_path = caller_path)
  source(resolved_path, local = local, ...)
}

# Choose a safe visualization sample size that does not exceed the number of
# available grouping levels.
#
# Args:
#   grouping_values: vector of length n_rows. Group identifiers used by a
#     plotting helper to sample random-effect groups.
#   requested_size: integer scalar. Preferred number of groups to visualize.
#
# Returns:
#   integer(1). Sample size clipped to the number of unique, non-missing
#   grouping levels, with a lower bound of 1 when at least one level exists.
safe_group_sample_size <- function(grouping_values, requested_size = 3L) {
  if (length(requested_size) != 1) {
    stop("requested_size must be a scalar.")
  }
  
  requested_size <- as.integer(requested_size)
  if (is.na(requested_size) || requested_size < 1L) {
    stop("requested_size must be a positive integer.")
  }
  
  available_levels <- unique(grouping_values[!is.na(grouping_values)])
  available_count <- length(available_levels)
  if (available_count < 1L) {
    stop("grouping_values must contain at least one non-missing level.")
  }
  
  as.integer(min(requested_size, available_count))
}
