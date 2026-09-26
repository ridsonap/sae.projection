library(testthat)

test_that("ma_projection works for linear regression and returns expected outputs", {
  data(df_svy22, package = "sae.projection")
  data(df_svy23, package = "sae.projection")

  df22_sub <- df_svy22[!is.na(df_svy22$income), ][1:500, ]
  df23_sub <- df_svy23[!is.na(df_svy23$income), ][1:1000, ]

  res <- ma_projection(
    formula = income ~ age + sex + edu + disability,
    cluster_ids = "PSU",
    weight = "WEIGHT",
    strata = "STRATA",
    domain = c("PROV", "REGENCY"),
    working_model = parsnip::linear_reg(),
    data_model = df22_sub,
    data_proj = df23_sub,
    nest = TRUE
  )

  expect_type(res, "list")
  expect_true(all(c("projection", "df_result", "working_model", "model", "prediction") %in% names(res)))
  expect_s3_class(res$projection, "data.frame")
  expect_true(all(c("ypr", "var_ypr", "rse_ypr") %in% names(res$projection)))
  expect_false(any(is.na(res$projection$ypr)))
  expect_true(all(res$projection$ypr > 0))
  expect_equal(res$projection, res$df_result)
})

test_that("ma_projection works for binary classification and outputs probabilities", {
  data(df_svy22, package = "sae.projection")
  data(df_svy23, package = "sae.projection")

  df22_neet <- df_svy22[dplyr::between(df_svy22$age, 15, 24), ][1:500, ]
  df23_neet <- df_svy23[dplyr::between(df_svy23$age, 15, 24), ][1:1000, ]

  res <- ma_projection(
    formula = neet ~ sex + edu + disability,
    cluster_ids = ~ PSU,
    weight = ~ WEIGHT,
    strata = ~ STRATA,
    domain = ~ PROV + REGENCY,
    working_model = parsnip::logistic_reg(),
    data_model = df22_neet,
    data_proj = df23_neet,
    nest = TRUE
  )

  expect_type(res, "list")
  expect_s3_class(res$projection, "data.frame")
  expect_false(any(is.na(res$projection$ypr)))
  # Probabilities must be bounded in [0, 1]
  expect_true(all(res$projection$ypr >= 0 & res$projection$ypr <= 1))
})

test_that("ma_projection handles domains absent in data_model without NA", {
  data(df_svy22, package = "sae.projection")
  data(df_svy23, package = "sae.projection")

  # Exclude REGENCY == 1 from data_model
  df22_sub <- df_svy22[!is.na(df_svy22$income) & df_svy22$REGENCY != 1, ][1:500, ]
  df23_sub <- df_svy23[!is.na(df_svy23$income), ][1:1000, ]

  res_no_bc <- ma_projection(
    income ~ age + sex + edu + disability,
    cluster_ids = "PSU", weight = "WEIGHT", strata = "STRATA",
    domain = c("PROV", "REGENCY"),
    working_model = parsnip::linear_reg(),
    data_model = df22_sub,
    data_proj = df23_sub,
    nest = TRUE,
    bias_correction = FALSE
  )

  expect_false(any(is.na(res_no_bc$projection$ypr)))
  expect_false(any(is.na(res_no_bc$projection$var_ypr)))

  res_bc <- ma_projection(
    income ~ age + sex + edu + disability,
    cluster_ids = "PSU", weight = "WEIGHT", strata = "STRATA",
    domain = c("PROV", "REGENCY"),
    working_model = parsnip::linear_reg(),
    data_model = df22_sub,
    data_proj = df23_sub,
    nest = TRUE,
    bias_correction = TRUE
  )

  expect_false(any(is.na(res_bc$projection$ypr)))
  expect_false(any(is.na(res_bc$projection$var_ypr)))
})

test_that("ma_projection works when domain is only in data_proj", {
  data(df_svy_A, package = "sae.projection")
  data(df_svy_B, package = "sae.projection")

  res <- ma_projection(
    formula = Y ~ x1 + x2 + x3,
    cluster_ids = "num",
    weight = "weight",
    domain = "regency",
    working_model = parsnip::logistic_reg(),
    data_model = df_svy_A,
    data_proj = df_svy_B
  )

  expect_type(res, "list")
  expect_s3_class(res$projection, "data.frame")
  expect_true("regency" %in% names(res$projection))
  expect_false(any(is.na(res$projection$ypr)))
})

test_that("ma_projection correctly aborts when required variables are missing", {
  data(df_svy_A, package = "sae.projection")
  data(df_svy_B, package = "sae.projection")

  expect_error(
    ma_projection(
      formula = Y ~ non_existent_x,
      cluster_ids = "num",
      weight = "weight",
      domain = "province",
      working_model = parsnip::logistic_reg(),
      data_model = df_svy_A,
      data_proj = df_svy_B
    )
  )

  expect_error(
    ma_projection(
      formula = non_existent_y ~ x1,
      cluster_ids = "num",
      weight = "weight",
      domain = "province",
      working_model = parsnip::logistic_reg(),
      data_model = df_svy_A,
      data_proj = df_svy_B
    ),
    regexp = "Target variable.*not found"
  )
})
