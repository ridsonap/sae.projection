#' Model-Assisted Projection Estimator
#'
#' @description The function addresses the problem of combining information from two or more independent surveys, a common challenge in survey sampling. It focuses on cases where: \cr
#' \itemize{
#'    \item **Survey 1:** A large sample collects only auxiliary information.
#'    \item **Survey 2:** A much smaller sample collects both the variables of interest and the auxiliary variables.
#' }
#' The function implements a model-assisted projection estimation method based on a working model. The working models that can be used include several machine learning models that can be seen in the details section.
#'
#' @references
#' \enumerate{
#'  \item Kim, J. K., & Rao, J. N. (2012). Combining data from two independent surveys: a model-assisted approach. Biometrika, 99(1), 85-100.
#' }
#'
#' @param formula A model formula. All variables used must exist in both \code{data_model} and \code{data_proj}.
#' @param cluster_ids Column name (character) or formula specifying cluster identifiers from highest to lowest level. Use \code{~0} or \code{~1} if there are no clusters.
#' @param weight Column name in \code{data_proj} (and \code{data_model}) representing the survey weights.
#' @param strata Column name for stratification; use \code{NULL} if no strata are used.
#' @param domain Character vector or formula specifying domain variable names. Must be present in \code{data_proj}.
#' @param summary_function A function to compute domain-level estimates (default: \code{"mean"}, \code{"total"}, \code{"variance"}).
#' @param working_model A parsnip model object specifying the working model (see \verb{@details}).
#' @param data_model Data frame (small sample, Survey 2) containing both target and auxiliary variables.
#' @param data_proj Data frame (large sample, Survey 1) containing auxiliary variables and domain variables.
#' @param model_metric A \code{yardstick::metric_set()} function, or \code{NULL} to use default metrics.
#' @param cv_folds Number of folds for k-fold cross-validation.
#' @param tuning_grid Either a data frame with tuning parameters or a positive integer specifying the number of grid search candidates.
#' @param parallel_over Specifies parallelization mode: \code{"resamples"}, \code{"everything"}, or \code{NULL}.
#' @param cores Integer specifying number of cores for parallel execution. If \code{NULL}, defaults to at most 2 cores.
#' @param seed Integer seed for reproducibility.
#' @param bias_correction Logical; if \code{TRUE}, applies domain-level bias correction from Survey 2 where available. If \code{FALSE} (default), uses the pure model-assisted projection estimator as formulated in Kim & Rao (2012).
#' @param return_yhat Logical; if \code{TRUE}, returns survey estimates of the target variable from \code{data_model}.
#' @param ... Additional arguments passed to \code{\link[survey]{svydesign}}.
#'
#' @return A list containing:
#'   \item{working_model}{The extracted fit engine from the final workflow.}
#'   \item{model}{Alias for \code{working_model}.}
#'   \item{prediction}{A vector of predicted values or probabilities on \code{data_proj}.}
#'   \item{projection}{A data frame with domain-level estimates.}
#'     \itemize{
#'       \item \code{domain} – Domain identifier(s).
#'       \item \code{ypr} – Projection estimator results for each domain.
#'       \item \code{var_ypr} – Estimated variance of the projection estimator.
#'       \item \code{rse_ypr} – Relative standard error (in \%).
#'     }
#'   \item{df_result}{Alias for \code{projection}.}
#'   \item{return_yhat}{(Optional) Survey estimates of the target variable from \code{data_model}, present when \code{return_yhat = TRUE}.}
#'
#' @export
#' @import bonsai
#' @import tidymodels
#' @import lightgbm
#' @import ranger
#'
#' @details
#' The following working models are supported via the \pkg{parsnip} interface:
#' \itemize{
#'   \item \code{linear_reg()} – Linear regression
#'   \item \code{logistic_reg()} – Logistic regression
#'   \item \code{linear_reg(engine = "stan")} – Bayesian linear regression
#'   \item \code{logistic_reg(engine = "stan")} – Bayesian logistic regression
#'   \item \code{poisson_reg()} – Poisson regression
#'   \item \code{decision_tree()} – Decision tree
#'   \item \code{nearest_neighbor()} – k-Nearest Neighbors (k-NN)
#'   \item \code{naive_bayes()} – Naive Bayes classifier
#'   \item \code{mlp()} – Multi-layer perceptron (neural network)
#'   \item \code{svm_linear()} – Support vector machine with linear kernel
#'   \item \code{svm_poly()} – Support vector machine with polynomial kernel
#'   \item \code{svm_rbf()} – Support vector machine with radial basis function (RBF) kernel
#'   \item \code{bag_tree()} – Bagged decision tree
#'   \item \code{bart()} – Bayesian Additive Regression Trees (BART)
#'   \item \code{rand_forest(engine = "ranger")} – Random forest (via ranger)
#'   \item \code{rand_forest(engine = "aorsf")} – Accelerated oblique random forest (AORF; Jaeger et al. 2022, 2024)
#'   \item \code{boost_tree(engine = "lightgbm")} – Gradient boosting (LightGBM)
#'   \item \code{boost_tree(engine = "xgboost")} – Gradient boosting (XGBoost)
#' }
#' For a complete list of supported models and engines, see \href{https://www.tmwr.org/pre-proc-table}{Tidy Modeling With R}.
#'
#' @examples
#' \dontrun{
#' library(sae.projection)
#' library(dplyr)
#' library(bonsai)
#'
#' df_svy22_income <- df_svy22 %>% filter(!is.na(income))
#' df_svy23_income <- df_svy23 %>% filter(!is.na(income))
#'
#' # Linear regression
#' lm_proj <- ma_projection(
#'   income ~ age + sex + edu + disability,
#'   cluster_ids = "PSU", weight = "WEIGHT", strata = "STRATA",
#'   domain = c("PROV", "REGENCY"),
#'   working_model = linear_reg(),
#'   data_model = df_svy22_income,
#'   data_proj = df_svy23_income,
#'   nest = TRUE
#' )
#'
#' df_svy22_neet <- df_svy22 %>%
#'      filter(between(age, 15, 24))
#' df_svy23_neet <- df_svy23 %>%
#'      filter(between(age, 15, 24))
#'
#' # Logistic regression
#' lr_proj <- ma_projection(
#'   formula = neet ~ sex + edu + disability,
#'   cluster_ids = "PSU",
#'   weight = "WEIGHT",
#'   strata = "STRATA",
#'   domain = c("PROV", "REGENCY"),
#'   working_model = logistic_reg(),
#'   data_model = df_svy22_neet,
#'   data_proj = df_svy23_neet,
#'   nest = TRUE
#' )
#' }
#' @md
ma_projection <- function(
    formula, cluster_ids, weight, strata = NULL, domain,
    summary_function = "mean", working_model,
    data_model, data_proj, model_metric = NULL,
    cv_folds = 3, tuning_grid = 10, parallel_over = "resamples",
    cores = NULL, seed = 1, bias_correction = FALSE,
    return_yhat = FALSE, ...) {

  # Variable validation ---------------------------------------------------
  cluster_ids <- .check_variable(cluster_ids, data_model, data_proj, model_required = TRUE)
  weight <- .check_variable(weight, data_model, data_proj, model_required = TRUE)
  strata <- .check_variable(strata, data_model, data_proj, model_required = TRUE)

  # Check domain variable(s)
  domain_chr <- domain
  if (methods::is(domain, "formula")) {
    domain_chr <- all.vars(domain)
  }
  has_domain_in_proj <- all(domain_chr %in% colnames(data_proj))
  if (!has_domain_in_proj) {
    diff_proj <- setdiff(domain_chr, colnames(data_proj))
    cli::cli_abort('domain variable "{diff_proj[1]}" is not found in the data_proj')
  }

  has_domain_in_model <- all(domain_chr %in% colnames(data_model))
  if (methods::is(domain, "character")) {
    domain <- stats::as.formula(paste0("~", paste(domain, collapse = " + ")))
  }

  # Target variable and problem mode --------------------------------------
  y_name <- as.character(rlang::f_lhs(formula))
  if (!y_name %in% colnames(data_model)) {
    cli::cli_abort('Target variable "{y_name}" is not found in data_model')
  }
  y <- data_model[[y_name]]

  # Determine problem type (regression or classification)
  model_mode <- working_model$mode
  if (!is.null(model_mode) && model_mode != "unknown") {
    type <- model_mode
  } else if (inherits(y, c("factor", "character", "logical"))) {
    type <- "classification"
  } else {
    type <- "regression"
  }

  if (type == "regression") {
    if (is.null(model_metric)) model_metric <- yardstick::metric_set(yardstick::rmse)
  } else {
    if (is.null(model_metric)) model_metric <- yardstick::metric_set(yardstick::f_meas)
    if (!is.factor(data_model[[y_name]])) {
      data_model[[y_name]] <- as.factor(data_model[[y_name]])
      y <- data_model[[y_name]]
    }
  }

  summary_function <- match.arg(summary_function, c("mean", "total", "variance"))
  if (summary_function == "mean") {
    FUN <- survey::svymean
  } else if (summary_function == "total") {
    FUN <- survey::svytotal
  } else if (summary_function == "variance") {
    FUN <- survey::svyvar
  } else {
    cli::cli_abort('summary_function must be "mean", "total", or "variance" not {summary_function}')
  }

  # Model specification and tuning ----------------------------------------
  all_tune <- grepl("tune", as.character(sapply(working_model[["args"]], rlang::quo_get_expr)))
  no_tune <- all(all_tune == FALSE)
  has_re <- has_random_effect(formula)

  if (has_re || no_tune) {
    model_wf <- parsnip::set_mode(working_model, mode = type)
    final_fit <- parsnip::fit(model_wf, formula, data_model)
  } else {
    model_spec <- parsnip::set_mode(working_model, mode = type)
    model_rec <- recipes::recipe(formula, data = data_model) %>%
      recipes::step_dummy(recipes::all_nominal_predictors()) %>%
      recipes::step_zv(recipes::all_predictors())

    model_wf <- workflows::workflow() %>%
      workflows::add_model(model_spec) %>%
      workflows::add_recipe(model_rec)

    # Set k-fold cross-validation
    set.seed(seed)
    if (type == "regression") {
      model_fold <- rsample::vfold_cv(data_model, v = cv_folds)
    } else {
      model_fold <- rsample::vfold_cv(data_model, v = cv_folds, strata = y_name)
    }

    if (is.null(cores)) {
      max_cores <- parallel::detectCores(logical = FALSE)
      if (is.na(max_cores) || max_cores < 1) max_cores <- 1
      cores <- min(2, max_cores)
    }

    cl <- parallel::makePSOCKcluster(cores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    doParallel::registerDoParallel(cl)

    set.seed(seed)
    tune_results <- model_wf %>%
      tune::tune_grid(
        resamples = model_fold,
        grid = tuning_grid,
        metrics = model_metric,
        control = tune::control_grid(parallel_over = parallel_over, verbose = FALSE, save_pred = TRUE, save_workflow = TRUE)
      )

    best_param <- tune::select_best(tune_results)
    final_fit <- tune::finalize_workflow(model_wf, best_param)
    final_fit <- parsnip::fit(final_fit, data_model)
  }

  final_model <- tune::extract_fit_engine(final_fit)

  # Prediction on Survey 1 and Survey 2 -----------------------------------
  if (type == "classification") {
    prob_proj <- stats::predict(final_fit, new_data = data_proj, type = "prob")
    prob_model <- stats::predict(final_fit, new_data = data_model, type = "prob")

    levs <- levels(as.factor(data_model[[y_name]]))
    pos_level <- if ("1" %in% levs) "1" else if ("Yes" %in% levs) "Yes" else levs[length(levs)]
    pos_col <- paste0(".pred_", pos_level)

    if (!pos_col %in% colnames(prob_proj)) {
      pos_col <- colnames(prob_proj)[ncol(prob_proj)]
    }

    data_proj$ypr <- prob_proj[[pos_col]]
    prediction <- prob_model[[pos_col]]

    y_num <- as.numeric(as.character(data_model[[y_name]]) == pos_level)
    data_model$bias <- y_num - prediction
  } else {
    data_proj$ypr <- stats::predict(final_fit, new_data = data_proj)[[1]]
    prediction <- stats::predict(final_fit, new_data = data_model)[[1]]
    data_model$bias <- y - prediction
  }

  # Survey 1 estimation (Kim & Rao 2012 Projection Estimator) -------------
  svy1_design <- survey::svydesign(
    ids = cluster_ids,
    weights = weight,
    strata = strata,
    data = data_proj,
    ...
  )

  est_ypr <- survey::svyby(
    formula = ~ypr,
    by = domain,
    design = svy1_design,
    FUN = FUN,
    vartype = c("var")
  )
  colnames(est_ypr)[colnames(est_ypr) == "var"] <- "var_ypr"

  # Bias correction and variance calculation ------------------------------
  if (has_domain_in_model) {
    svy2_design <- survey::svydesign(
      ids = cluster_ids,
      weights = weight,
      strata = strata,
      data = data_model,
      ...
    )
    est_bias <- survey::svyby(
      formula = ~bias,
      by = domain,
      design = svy2_design,
      FUN = FUN,
      vartype = c("var")
    )
    colnames(est_bias)[colnames(est_bias) == "var"] <- "var_bias"

    df_result <- dplyr::left_join(est_ypr, est_bias, by = domain_chr)
    # Handle domains not sampled in Survey 2
    df_result$bias[is.na(df_result$bias)] <- 0
    df_result$var_bias[is.na(df_result$var_bias)] <- 0

    if (bias_correction) {
      ypr_final <- df_result$ypr + df_result$bias
    } else {
      ypr_final <- df_result$ypr
    }
    var_ypr_final <- df_result$var_ypr + df_result$var_bias
    df_result[, c("bias", "var_bias")] <- NULL
  } else {
    if (bias_correction) {
      cli::cli_warn("Domain variable(s) not found in data_model. bias_correction = TRUE ignored; using pure projection estimator.")
    }
    df_result <- est_ypr
    ypr_final <- df_result$ypr
    var_ypr_final <- df_result$var_ypr
  }

  rse_ypr <- ifelse(ypr_final != 0, sqrt(pmax(0, var_ypr_final)) * 100 / ypr_final, 0)

  df_result$ypr <- ypr_final
  df_result$var_ypr <- var_ypr_final
  df_result$rse_ypr <- rse_ypr

  all_result <- list(
    working_model = final_model,
    model = final_model,
    prediction = data_proj$ypr,
    projection = df_result,
    df_result = df_result
  )

  # Return direct estimates of y on Survey 2 if requested ----------------
  if (return_yhat) {
    if (has_domain_in_model) {
      if (!exists("svy2_design", inherits = FALSE)) {
        svy2_design <- survey::svydesign(
          ids = cluster_ids,
          weights = weight,
          strata = strata,
          data = data_model,
          ...
        )
      }
      est_y <- survey::svyby(
        formula = stats::as.formula(paste0("~", y_name)),
        by = domain,
        design = svy2_design,
        FUN = FUN,
        vartype = c("var", "cvpct")
      )
      colnames(est_y)[colnames(est_y) == "cv%"] <- "rse"
      all_result$return_yhat <- est_y
    } else {
      cli::cli_warn("return_yhat = TRUE ignored because domain is not present in data_model.")
    }
  }

  return(all_result)
}


.check_variable <- function(variable, data_model, data_proj, model_required = TRUE) {
  if (is.null(variable)) {
    return(variable)
  }

  if (methods::is(variable, "formula")) {
    f_str <- paste(deparse(variable), collapse = " ")
    if (grepl("~\\s*0", f_str) || grepl("~\\s*1", f_str)) return(variable)

    if (model_required) {
      tryCatch({
        dat <- stats::model.frame(variable, data_model, na.action = NULL)
      },
      error = function(x) {
        cli::cli_abort('variable "{variable}" is not found in the data_model')
      })
    }

    tryCatch({
      dat <- stats::model.frame(variable, data_proj, na.action = NULL)
    },
    error = function(x) {
      cli::cli_abort('variable "{variable}" is not found in the data_proj')
    })
    return(variable)

  } else if (methods::is(variable, "character")) {
    if (model_required) {
      diff_model <- setdiff(variable, colnames(data_model))
      if (length(diff_model) > 0) {
        cli::cli_abort('variable "{diff_model[1]}" is not found in the data_model')
      }
    }
    diff_proj <- setdiff(variable, colnames(data_proj))
    if (length(diff_proj) > 0) {
      cli::cli_abort('variable "{diff_proj[1]}" is not found in the data_proj')
    }

    if (length(variable) > 1) {
      return(stats::as.formula(paste0("~", paste0(variable, collapse = " + "))))
    } else {
      return(stats::as.formula(paste0("~", variable)))
    }
  } else {
    cli::cli_abort('variable "{variable}" must be character or formula')
  }
}


has_random_effect <- function(formula) {
  any(grepl("\\|", deparse(formula)))
}
