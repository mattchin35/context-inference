# Shared data preprocessing utilities

convertNoneToNA <- function(df, cols) {
  if (is.character(cols) && length(cols) == 1) {
    cols <- c(cols)
  }
  
  naVals <- c("None", "none", "NULL", "null", "", "N/A", "NA")
  
  for (col in cols) {
    if (col %in% names(df)) {
      df[[col]] <- ifelse(df[[col]] %in% naVals, NA, df[[col]])
    }
  }
  
  return(df)
}

removeNARows <- function(df, cols, mode = "any") {
  if (is.character(cols) && length(cols) == 1) {
    cols <- c(cols)
  }
  
  naMatrix <- is.na(df[, cols, drop = FALSE])
  
  if (mode == "any") {
    keepRows <- rowSums(naMatrix) == 0
  } else if (mode == "all") {
    keepRows <- !(rowSums(naMatrix) == length(cols))
  } else {
    stop("mode must be 'any' or 'all'")
  }
  
  return(df[keepRows, ])
}

removeRowsByColumn <- function(dataFrame, columnName, condition, verbose = TRUE) {
  if (!is.data.frame(dataFrame)) {
    stop("Input must be a dataframe")
  }
  
  if (!columnName %in% names(dataFrame)) {
    stop("Column '", columnName, "' not found in dataframe")
  }
  
  if (missing(condition)) {
    stop("Condition must be specified")
  }
  
  rowsBefore <- nrow(dataFrame)
  conditionMask <- condition(dataFrame[[columnName]])
  rowsToRemove <- sum(conditionMask, na.rm = TRUE)
  
  filteredData <- dataFrame[!conditionMask, ]
  rownames(filteredData) <- NULL
  
  if (verbose) {
    cat("Column analyzed: '", columnName, "'\n", sep = "")
    cat("Rows removed:", rowsToRemove, "\n")
    cat("Rows remaining:", nrow(filteredData), "\n")
    cat("Percentage kept:",
        round(nrow(filteredData) / rowsBefore * 100, 1), "%\n")
  }
  
  return(filteredData)
}

stack_dataframes <- function(..., .id = NULL) {
  df_list <- list(...)
  
  if (!is.null(.id)) {
    result <- dplyr::bind_rows(df_list, .id = .id)
  } else {
    result <- dplyr::bind_rows(df_list)
  }
  
  cat("Stacked", length(df_list), "dataframes\n")
  cat("Total rows:", nrow(result), "\n")
  cat("Columns:", paste(names(result), collapse = ", "), "\n")
  
  return(result)
}