workspace <- Sys.getenv("AQUACULTURE_WORKSPACE")
if (!nzchar(workspace)) stop("Run this workflow through Scripts/run.py.")
project_path <- function(relative) file.path(workspace, relative)
external_path <- function(relative) {
  root <- Sys.getenv("AQUACULTURE_EXTERNAL_DOWNLOAD")
  if (!nzchar(root)) stop("External data directory has not been configured.")
  file.path(root, sub("^download/?", "", relative))
}
require_installed_packages <- function(packages, ...) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop(paste("Install required R packages explicitly before running:", paste(missing, collapse = ", ")))
}
