

{
## EKERE JOHN ADACHE
## ID 2123522
## TOPIC: FRAUD DETECTION IN E-COMMERCE USING MACHINE LEARNING; A COMPARATIVE TEST BETWEEN FOUR ML ALGORITHMS

# Installing necessary package
install.packages("dplyr")

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Load your dataset
Data <- read.csv("C:\\Users\\localuser\\OneDrive - University of Bolton\\Documents\\Fraud.csv")

View(Data)

# Summary statistics
summary(Data)
str(Data)

# Checking dataset dimensions
num_rows <- nrow(Data)
num_columns <- ncol(Data)
cat("Number of rows:", num_rows, "\n")
cat("Number of columns:", num_columns, "\n")

column_names <- names(Data)

# Print the column names
print(column_names)

# Checking uniqueness
unique_check <- sapply(Data, function(col) length(unique(col)) == length(col))

# Print the result
print(unique_check)

# Get the names of columns with unique values
unique_column_names <- names(unique_check[unique_check])

# Print the result
print(unique_column_names)

# Checking missing values in dataframe
missing_values <- colSums(is.na(Data))

# Print the missing values count for each column
print(missing_values)

library(dplyr)

# Removing specified columns
data_clean <- Data %>%
  select(-nameOrig, -nameDest, -isFlaggedFraud)

View(data_clean)

# Set a seed for reproducibility
set.seed(123)

# Randomly sample 5000 rows with an equal mix of 0s and 1s
fraudulent_rows <- data_clean[data_clean$isFraud == 1, ]
non_fraudulent_rows <- data_clean[data_clean$isFraud == 0, ]

# Sample 2500 rows from each category
sampled_fraudulent <- fraudulent_rows[sample(nrow(fraudulent_rows), 2500), ]
sampled_non_fraudulent <- non_fraudulent_rows[sample(nrow(non_fraudulent_rows), 2500), ]

# Combine the sampled rows
sampled_data <- rbind(sampled_fraudulent, sampled_non_fraudulent)

# Shuffle the rows to mix the 0s and 1s
sampled_data <- sampled_data[sample(nrow(sampled_data)), ]

# Verify the class balance
table(sampled_data$isFraud)

View(sampled_data)

# Checking 'sampled_data' dataframe
summary_stats <- summary(sampled_data)
print(summary_stats)

# Histogram
hist(sampled_data$amount, main = "Histogram of Amount", xlab = "Amount")

# Boxplot
boxplot(sampled_data$amount, main = "Box Plot of Amount")

# Scattered plot of 'sampled_data' showing 'amount' and 'oldbalanceOrg' because they are numerical variables
plot(sampled_data$amount, sampled_data$oldbalanceOrg, main = "Scatter Plot of Amount vs. Old Balance",
     xlab = "Amount", ylab = "Old Balance")

# Histogram for 'oldbalanceOrg'
hist(sampled_data$oldbalanceOrg, main = "Histogram of Old Balance Orig", xlab = "Old Balance Orig")

# Box plot for 'newbalanceOrig'
boxplot(sampled_data$newbalanceOrig, main = "Box Plot of New Balance Orig")

# Bar plot for 'type'
barplot(table(sampled_data$type), main = "Bar Plot of Transaction Types", xlab = "Transaction Type")

# Check 'amount' values for 0s and 1s in the 'isFraud' column
amount_for_0 <- sampled_data$amount[sampled_data$isFraud == 0]
amount_for_1 <- sampled_data$amount[sampled_data$isFraud == 1]

# Print or visualize the 'amount' values for 0s and 1s
hist(amount_for_0, main = "Histogram of Amount for Non-Fraudulent Transactions", xlab = "Amount")
hist(amount_for_1, main = "Histogram of Amount for Fraudulent Transactions", xlab = "Amount")

# Create a table of 'amount' values for 0s and 1s in the 'isFraud' column
amount_table <- table(sampled_data$isFraud, sampled_data$amount)

# Print the table
print(amount_table)

View(amount_table)

# Define the 'amount' ranges
amount_ranges <- cut(sampled_data$amount, breaks = c(0, 1000, 5000, 10000, max(sampled_data$amount)))

# Create a table that summarizes the counts of 0s and 1s for each 'amount' range
amount_fraud_summary <- table(amount_ranges, sampled_data$isFraud)

# Print the summary table
print(amount_fraud_summary)

# Create a binary feature 'high_amount_fraud' based on the 'amount' range
sampled_data$high_amount_fraud <- ifelse(sampled_data$amount > 10000 & sampled_data$amount <= 2.13e+07, 1, 0)

# Check the distribution of the new feature
table(sampled_data$high_amount_fraud)

# Check the data types of all columns in your dataframe
column_data_types <- sapply(sampled_data, class)

# Identify which columns are categorical or non-numeric
categorical_columns <- names(sampled_data)[sapply(sampled_data, is.factor)]
binary_columns <- names(sampled_data)[sapply(sampled_data, function(x) length(unique(x)) == 2)]

# Print the data types and categorical/binary columns
print(column_data_types)
print("Categorical Columns:")
print(categorical_columns)
print("Binary Columns:")
print(binary_columns)

# Performing one-hot encoding for the 'type' variable
sampled_data_encoded <- as.data.frame(model.matrix(~ type - 1, data = sampled_data))

# Combine the encoded data with the original dataframe while excluding the original 'type' column
sampled_data <- cbind(sampled_data[, -which(names(sampled_data) == "type")], sampled_data_encoded)

# Print the first few rows of the updated dataframe to verify the encoding
head(sampled_data)

# Calculate the correlation matrix
correlation_matrix <- cor(sampled_data)

# Find the features that are highly correlated with the target variable 'isFraud'
correlation_with_target <- abs(correlation_matrix[,"isFraud"])
highly_correlated_features <- names(correlation_with_target[correlation_with_target > 0.1])

library(corrplot)
corr_matrix <- cor(sampled_data[, -which(names(sampled_data) == "isFraud")])
corrplot(corr_matrix, method = "color")

# Print the highly correlated features
print("Highly Correlated Features:")
print(highly_correlated_features)

install.packages("caret")
library(caret)

# Splitting data into training and testing sets
set.seed(123)  # for reproducibility
split_index <- createDataPartition(sampled_data$isFraud, p = 0.7, list = FALSE)
train_data <- sampled_data[split_index, ]
test_data <- sampled_data[-split_index, ]

# Assuming 'train_data' contains your training dataset
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- rfe(train_data[, -which(names(train_data) == "isFraud")], train_data$isFraud, sizes = c(1:10), rfeControl = control)
print(results)

# Define a range of thresholds (0 to 1)
thresholds <- seq(0, 1, by = 0.01)
# Initialize variables to store F1-scores and other metrics
f1_scores <- numeric(length(thresholds))
precision <- numeric(length(thresholds))
recall <- numeric(length(thresholds))
# Convert rf_predictions to numeric
rf_numeric <- as.numeric(as.character(rf_predictions))

# Calculate F1-scores for each threshold
for (i in 1:length(thresholds)) {
  # Apply the threshold to model predictions
  thresholded_predictions <- ifelse(rf_numeric > thresholds[i], 1, 0)
  
  # Calculate precision and recall
  precision[i] <- sum(thresholded_predictions == 1 & test_data$isFraud == 1) / sum(thresholded_predictions == 1)
  recall[i] <- sum(thresholded_predictions == 1 & test_data$isFraud == 1) / sum(test_data$isFraud == 1)
  
  # Calculate F1-score
  f1_scores[i] <- 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
}
# Find the threshold that maximizes the F1-score
optimal_threshold <- thresholds[which.max(f1_scores)]
# Visualize the F1-score vs. threshold
plot(thresholds, f1_scores, type = "l", xlab = "Threshold", ylab = "F1-Score", main = "Threshold Optimization")
# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply the optimal threshold to predictions
optimal_predictions <- ifelse(rf_numeric > optimal_threshold, 1, 0)

# since 'rf_model' is my trained Random Forest model
var_importance <- importance(rf_model)
print(var_importance)

#### second model
# Load the required library for Decision Trees
library(rpart)

# Train the Decision Tree model
tree_model <- rpart(isFraud ~ ., data = train_data, method = "class")

# Make predictions on the test data
tree_predictions <- predict(tree_model, test_data, type = "class")

# Evaluate the Decision Tree model's performance
tree_confusion_matrix <- table(Actual = test_data$isFraud, Predicted = tree_predictions)
print("Decision Tree Confusion Matrix:")
print(tree_confusion_matrix)

# Calculate accuracy for the Decision Tree model
tree_correct_predictions <- sum(tree_predictions == test_data$isFraud)
tree_total_predictions <- nrow(test_data)
tree_accuracy <- tree_correct_predictions / tree_total_predictions
print(paste("Decision Tree Accuracy:", tree_accuracy))

# Calculate F1-score for the Decision Tree model (similar to what you did for the Random Forest and SVM)
tree_f1_score <- 2 * (sum(tree_predictions == 1 & test_data$isFraud == 1) /
                        sum(tree_predictions == 1) * sum(test_data$isFraud == 1)) /
  (sum(tree_predictions == 1 & test_data$isFraud == 1) +
     sum(tree_predictions == 1) + sum(test_data$isFraud == 1))
print(paste("Decision Tree F1-Score:", tree_f1_score))

# Check the distribution of values in the isFraud column
table(train_data$isFraud)

# Check for missing values in the isFraud column
sum(is.na(train_data$isFraud))

# Check the data type of the isFraud column
class(train_data$isFraud)

# Check the class distribution of isFraud (if class imbalance is suspected)
prop.table(table(train_data$isFraud))

unique(train_data$isFraud)

#### third model
# Load the required library for SVM
library(e1071)  # This library provides an interface to the LIBSVM library

# Train the SVM model
svm_model <- svm(isFraud ~ ., data = train_data, kernel = "linear", cost = 1)

# Make predictions on the test data
svm_predictions <- predict(svm_model, test_data)

# Evaluate the model's performance
svm_confusion_matrix <- table(Actual = test_data$isFraud, Predicted = svm_predictions)
print(svm_confusion_matrix)

# Calculate accuracy for the SVM model
svm_correct_predictions <- sum(svm_predictions == test_data$isFraud)
svm_total_predictions <- nrow(test_data)
svm_accuracy <- svm_correct_predictions / svm_total_predictions
print(paste("SVM Accuracy:", svm_accuracy))

# Calculate F1-score for the SVM model 
precision_svm <- sum(svm_predictions == 1 & test_data$isFraud == 1) / sum(svm_predictions == 1)
recall_svm <- sum(svm_predictions == 1 & test_data$isFraud == 1) / sum(test_data$isFraud == 1)
f1_score_svm <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)
print(paste("SVM F1-Score:", f1_score_svm))

#### models comparison
# Load necessary libraries
library(randomForest)
library(e1071)
library(rpart)
library(class)

# considering I have my training and testing datasets ready
# train_data and test_data
# Defining the target variable
target_var <- "isFraud"
# Define the predictors
predictors <- setdiff(names(train_data), target_var)
# Create an empty dataframe to store model evaluation results
model_results <- data.frame(Model = character(0), Accuracy = numeric(0), F1_Score = numeric(0))
# Function to evaluate a model and add results to the dataframe
evaluate_model <- function(model_name, model, train_data, test_data) {
  # Train the model
  if (model_name == "Random Forest") {
    model <- randomForest(
      formula = as.formula(paste(target_var, "~", paste(predictors, collapse = "+"))),
      data = train_data,
      ntree = 100,  
      mtry = sqrt(length(predictors)),
      importance = TRUE
    )
  } else if (model_name == "SVM") {
    model <- svm(isFraud ~ ., data = train_data, kernel = "linear", cost = 1)
  } else if (model_name == "Decision Tree") {
    model <- rpart(isFraud ~ ., data = train_data, method = "class")
  }
  # Make predictions on the test data
  predictions <- predict(model, test_data[, predictors])
  # Calculate accuracy
  correct_predictions <- sum(predictions == test_data[, target_var])
  total_predictions <- nrow(test_data)
  accuracy <- correct_predictions / total_predictions
  # Calculate F1-score
  precision <- sum(predictions == 1 & test_data[, target_var] == 1) / sum(predictions == 1)
  recall <- sum(predictions == 1 & test_data[, target_var] == 1) / sum(test_data[, target_var] == 1)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  # Add results to the dataframe
  model_results <<- rbind(model_results, data.frame(Model = model_name, Accuracy = accuracy, F1_Score = f1_score))
}
# Evaluate each model and add results to the dataframe
evaluate_model("Random Forest", NULL, train_data, test_data)
evaluate_model("SVM", NULL, train_data, test_data)
evaluate_model("Decision Tree", NULL, train_data, test_data)

# Print the results
print(model_results)
# Visualize the results
library(ggplot2)
ggplot(model_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") + labs(title = "Model Comparison - Accuracy",
                                     x = "Model",  y = "Accuracy") +
  theme_minimal()
ggplot(model_results, aes(x = Model, y = F1_Score, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison - F1-Score",
       x = "Model",     y = "F1-Score") +
  theme_minimal()
}
