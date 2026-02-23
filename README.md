Car Price Prediction: Deep Learning vs. Traditional Regression

This repository explores vehicle price estimation using TensorFlow and Scikit-Learn. The project compares a Deep Learning approach (Neural Networks) against traditional statistical models (Ordinary Least Squares and Stochastic Gradient Descent).

📌 Project Overview
The goal is to predict the Present_Price of a vehicle based on various attributes. The project is split into two experimental setups:

Rnu.py: A standard approach including all available numerical and categorical features.

Run_exclude_Selling_Price.py: A more challenging scenario where Selling_Price is removed to test how well the model predicts the car's value based solely on its specifications and usage.

🛠️ Technical Workflow
1. Data Preprocessing
One-Hot Encoding: Converts categorical variables (Car Name, Fuel Type, Seller Type, Transmission) into numerical dummy variables.

Missing Values: Automatically detects and drops rows with missing data to ensure model stability.

Feature Scaling: Utilizes MinMaxScaler to normalize data between 0 and 1, which is critical for the convergence of the Neural Network and SGD models.

2. Model Architectures
The project evaluates three distinct modeling techniques:

TensorFlow Neural Network (MLP):

Input layer tailored to the feature count.

Two hidden layers (100 and 50 neurons) with ReLU activation.

Output layer with 1 neuron for regression.

📊 Evaluation & Visualization
Each script generates a three-panel comparison plot using matplotlib. These plots visualize the correlation between Actual Prices and Predicted Prices across:

Training Set

Testing Set

Validation Set
Linear Regression: A classic Scikit-Learn implementation using the Least Squares method.

SGD Regressor: An iterative optimization approach using Stochastic Gradient Descent, ideal for larger or high-dimensional datasets.
