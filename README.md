# House Price Prediction with MLflow

This project is a learning exercise focused on using [MLflow](https://mlflow.org/) for tracking and managing machine learning experiments. Through this project, I explored how MLflow helps track parameters, metrics, and models for better experimentation, model comparison, and reproducibility. The project involves training a machine learning model to predict California house prices using the California Housing dataset, with a strong focus on using MLflow to manage the end-to-end workflow.

## Table of Contents
- [Purpose](#purpose)
- [Learning Goals](#learning-goals)
- [Dataset](#dataset)
- [MLflow Workflow](#mlflow-workflow)
- [Usage](#usage)
- [Results](#results)

## Purpose

The primary purpose of this project is to get hands-on experience with MLflow, focusing on:
- Experiment tracking, including logging hyperparameters, metrics, and model artifacts.
- Model comparison across different hyperparameter configurations.
- Using the MLflow UI to review and select the best model run.
- Registering and storing models for potential deployment, using MLflow’s model registry.

## Learning Goals

1. **Understand Experiment Tracking**: Learn to track various experiment parameters and results over multiple runs using the MLflow interface.
2. **Implement Hyperparameter Tuning with MLflow**: Use `GridSearchCV` for hyperparameter tuning, and log the resulting configurations and model performance metrics.
3. **Explore the MLflow UI**: Use the MLflow UI to view, compare, and analyze experiment runs.
4. **Model Registry**: Practice saving and registering the best model for future reference, enabling easy retrieval and deployment from MLflow's model registry.

## Dataset

The [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) was chosen for its straightforward structure, making it ideal for exploring MLflow tracking without complex preprocessing. The dataset provides features like median income, house age, average rooms, and population, with house price as the target variable.

## MLflow Workflow

### 1. Data Preparation
   - The California Housing dataset is loaded and converted into a DataFrame.
   - Data is split into training and testing sets for model validation.

### 2. Model Training and Hyperparameter Tuning
   - A RandomForestRegressor model is trained using `GridSearchCV` to identify the best hyperparameters.
   - The following hyperparameters are tuned:
     - `n_estimators`: Number of trees.
     - `max_depth`: Maximum depth of trees.
     - `min_samples_split`: Minimum samples to split an internal node.
     - `min_samples_leaf`: Minimum samples at a leaf node.

### 3. Experiment Tracking with MLflow
   - **Parameters**: The best hyperparameters identified by `GridSearchCV` are logged to MLflow.
   - **Metrics**: Evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²) are logged to MLflow.
   - **Model**: The best model is logged and registered with MLflow for easy retrieval and future comparison.

### 4. Model Registry
   - MLflow’s model registry is used to store the best model, which helps track its version and provides easy access for future experiments and potential deployment.

## Usage

1. **Run the notebook**: Execute the notebook to load data, train the model, and log experiments with MLflow.
2. **Explore MLflow UI**: Launch the MLflow UI to examine experiment runs, compare metrics, and view saved models.
3. **Review and Select the Best Model**: Use MLflow’s comparison features to choose the optimal model for registration.

## Results

The project’s results, including the best model configuration and metrics, are accessible in the MLflow UI. This experiment demonstrates MLflow's power in organizing and analyzing machine learning workflows, helping with:
- Transparent tracking of model training and performance.
- Ease of model comparison and selection.
- Centralized storage and versioning of models, aiding reproducibility and deployment readiness.

## License

This project is open-source and licensed under the MIT License.
