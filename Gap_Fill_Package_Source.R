install.packages("devtools")
install.packages("roxygen2")
install.packages("xgboost")
install.packages("caret")
install.packages("dplyr")


library(dplyr)
library(xgboost)
library(caret)

# Define functions
add_time_vars <- function(data, date_col) {
  data[[date_col]] <- as.POSIXct(data[[date_col]])
  data$Month <- as.numeric(format(data[[date_col]], "%m"))
  data$Hour <- as.numeric(format(data[[date_col]], "%H")) + as.numeric(format(data[[date_col]], "%M")) / 60
  data$Month_sin <- sin((data$Month - 1) * (2 * pi / 12))
  data$Month_cos <- cos((data$Month - 1) * (2 * pi / 12))
  data$Hour_sin <- sin(data$Hour * (2 * pi / 24))
  data$Hour_cos <- cos(data$Hour * (2 * pi / 24))
  data$Time <- seq_len(nrow(data)) - 1
  return(data)
}

org_data <- function(data, x_cols, y_col) {
  data <- data[!is.na(data[[y_col]]), ]
  X <- data[, x_cols, drop = FALSE]
  y <- data[[y_col]]
  list(X = X, y = y)
}

optimize_hyperparameters <- function(data, x_cols, y_col) {
  organized_data <- org_data(data, x_cols, y_col)
  X <- organized_data$X
  y <- organized_data$y
  
  # Convert to matrix
  X_matrix <- as.matrix(X)
  
  # Define parameter grid with required columns
  param_grid <- expand.grid(
    nrounds = 100,
    eta = 0.1,
    max_depth = c(3, 5, 7),
    gamma = 0,
    colsample_bytree = c(0.8, 1),
    min_child_weight = c(1, 5),
    subsample = c(0.8, 1)
  )
  
  # Train model with caret
  control <- trainControl(method = "cv", number = 5)
  
  model <- train(
    X_matrix, y,
    method = "xgbTree",
    trControl = control,
    tuneGrid = param_grid,
    verbose = FALSE
  )
  
  return(model$bestTune)
}

cv_preds <- function(data, x_cols, y_col, hyperparams) {
  X <- data[, x_cols, drop = FALSE]
  y <- data[[y_col]]
  
  # Convert to matrix
  X_matrix <- as.matrix(X)
  
  # Fit model
  model <- xgboost(
    data = X_matrix, label = y, nrounds = 100,
    objective = "reg:squarederror",
    params = list(
      colsample_bytree = hyperparams$colsample_bytree,
      max_depth = hyperparams$max_depth,
      min_child_weight = hyperparams$min_child_weight,
      subsample = hyperparams$subsample
    ),
    verbose = 0
  )
  
  # Predictions
  y_pred <- predict(model, X_matrix)
  data$predicted <- y_pred
  
  return(data)
}
# Example usage

# Load data
data <- read.csv("C:/Users/Chris/Downloads/US-Elm_2017.csv")

# Prompt user for column names
date_col <- "TIMESTAMP_START"
x_cols <- c("TA_F")
y_col <- "FC"

# Debugging prints to confirm input
print(paste("Date column: ", date_col))
print(paste("Feature columns: ", paste(x_cols, collapse = ", ")))
print(paste("Target column: ", y_col))

# Add time variables
data <- add_time_vars(data, date_col)

# Optimize hyperparameters
best_params <- optimize_hyperparameters(data, x_cols, y_col)
print(best_params)

# Gapfill predictions using cross-validation
data_with_preds <- cv_preds(data, x_cols, y_col, best_params)
print(head(data_with_preds))
