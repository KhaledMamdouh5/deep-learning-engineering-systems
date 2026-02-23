import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd


## Linear Regression Using NN TF
data = pd.read_csv(r"car data.csv")

print("Initial data info:")
data.info()


# Handle potential missing values (simple dropna, if any)
if data.isnull().sum().any():
    print("\nFound missing values. Dropping rows with NaNs.")
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

data = data.drop(columns=['Selling_Price'])
categorical_cols = data.select_dtypes(include=['object']).columns
# Now, 'Car_Name' will be in categorical_cols if it's of object type
print(f"\nCategorical columns to be one-hot encoded: {list(categorical_cols)}")
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("\nData info after one-hot encoding (including Car_Name):")
data.info() # Observe the significant increase in columns

# target = ['Y house price of unit area'] # Original target
target = ['Present_Price'] # Modified target for "car data.csv"
X = data.drop(columns=target)
Y = data[target[:]]

print("\nFeatures (X) info:")
X.info()
print("\nTarget (Y) info:")
Y.info()

X = X.values
Y = Y.values

print("\nShape of X and Y after preprocessing (with Car_Name OHE):", X.shape, Y.shape)
print(f"WARNING: High number of features ({X.shape[1]}) relative to samples ({X.shape[0]}). This can lead to overfitting.")


x_train, x_test_val, y_train, y_test_val = train_test_split(X, Y, test_size=0.3, random_state=50)
x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.6, random_state=50)

print("\nShapes of splits:")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:", x_test.shape, "y_test:", y_test.shape)
print("x_val:", x_val.shape, "y_val:", y_val.shape)

print("\nhi") # Original debug print
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)
x_val = min_max_scaler.transform(x_val)

## --- TensorFlow Neural Network Section ---
print("\n--- Training TensorFlow Neural Network (with Car_Name OHE) ---")
model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)), # Input shape now reflects many more features
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# Consider reducing epochs or using early stopping if overfitting is severe
history = model_tf.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_val, y_val), verbose=1)

loss_tf, mae_tf = model_tf.evaluate(x_val, y_val, verbose=0)
print(f"TensorFlow NN - Validation Loss (MSE): {loss_tf:.4f}, Validation MAE: {mae_tf:.4f}")

predictions_tf_test = model_tf.predict(x_test)
predictions_tf_train = model_tf.predict(x_train)
predictions_tf_val = model_tf.predict(x_val)

mse_tf_test = mean_squared_error(y_test, predictions_tf_test)
print(f"TensorFlow NN - Test MSE: {mse_tf_test:.4f}")


print("\nx_train shape for TF plot:", x_train.shape, "y_train shape for TF plot:", y_train.shape)

fig_tf, axes_tf = plt.subplots(1, 3, figsize=(18, 5))
fig_tf.suptitle("TensorFlow NN (Car_Name OHE): Actual vs. Predicted Present Price", fontsize=16)

axes_tf[0].scatter(x=y_train, y=predictions_tf_train, alpha=0.6)
axes_tf[0].set_xlabel("Actual Present Price", fontsize=10)
axes_tf[0].set_ylabel("Predicted Present Price",  fontsize=10)
axes_tf[0].set_title("Training Set")
min_max_train = [min(y_train.min(), predictions_tf_train.min()), max(y_train.max(), predictions_tf_train.max())]
axes_tf[0].plot(min_max_train, min_max_train, color='red', linestyle='--')

axes_tf[1].scatter(x=y_test, y=predictions_tf_test, alpha=0.6)
axes_tf[1].set_xlabel("Actual Present Price", fontsize=10)
axes_tf[1].set_ylabel("Predicted Present Price",  fontsize=10)
axes_tf[1].set_title("Testing Set")
min_max_test = [min(y_test.min(), predictions_tf_test.min()), max(y_test.max(), predictions_tf_test.max())]
axes_tf[1].plot(min_max_test, min_max_test, color='red', linestyle='--')

axes_tf[2].scatter(x=y_val, y=predictions_tf_val, alpha=0.6)
axes_tf[2].set_xlabel("Actual Present Price", fontsize=10)
axes_tf[2].set_ylabel("Predicted Present Price",  fontsize=10)
axes_tf[2].set_title("Validation Set")
min_max_val = [min(y_val.min(), predictions_tf_val.min()), max(y_val.max(), predictions_tf_val.max())]
axes_tf[2].plot(min_max_val, min_max_val, color='red', linestyle='--')

fig_tf.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## --- Scikit-Learn Linear Regression (Least Squares) Section ---
print("\n--- Training Scikit-Learn Linear Regression (Least Squares, with Car_Name OHE) ---")
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

predictions_lr_test = model_lr.predict(x_test)
mse_lr_test = mean_squared_error(y_test, predictions_lr_test)
print(f"Sklearn Linear Regression (Least Squares) - Test MSE: {mse_lr_test:.4f}")

predictions_lr_train = model_lr.predict(x_train)
predictions_lr_val = model_lr.predict(x_val)

fig_lr, axes_lr = plt.subplots(1, 3, figsize=(18, 5))
fig_lr.suptitle("Scikit-Learn Linear Regression (Least Squares, Car_Name OHE): Actual vs. Predicted", fontsize=16)

axes_lr[0].scatter(x=y_train, y=predictions_lr_train, alpha=0.6)
axes_lr[0].set_xlabel("Actual Present Price", fontsize=10)
axes_lr[0].set_ylabel("Predicted Present Price",  fontsize=10)
axes_lr[0].set_title("Training Set")
min_max_train_lr = [min(y_train.min(), predictions_lr_train.min()), max(y_train.max(), predictions_lr_train.max())]
axes_lr[0].plot(min_max_train_lr, min_max_train_lr, color='red', linestyle='--')

axes_lr[1].scatter(x=y_test, y=predictions_lr_test, alpha=0.6)
axes_lr[1].set_xlabel("Actual Present Price", fontsize=10)
axes_lr[1].set_ylabel("Predicted Present Price",  fontsize=10)
axes_lr[1].set_title("Testing Set")
min_max_test_lr = [min(y_test.min(), predictions_lr_test.min()), max(y_test.max(), predictions_lr_test.max())]
axes_lr[1].plot(min_max_test_lr, min_max_test_lr, color='red', linestyle='--')

axes_lr[2].scatter(x=y_val, y=predictions_lr_val, alpha=0.6)
axes_lr[2].set_xlabel("Actual Present Price", fontsize=10)
axes_lr[2].set_ylabel("Predicted Present Price",  fontsize=10)
axes_lr[2].set_title("Validation Set")
min_max_val_lr = [min(y_val.min(), predictions_lr_val.min()), max(y_val.max(), predictions_lr_val.max())]
axes_lr[2].plot(min_max_val_lr, min_max_val_lr, color='red', linestyle='--')

fig_lr.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## --- Scikit-Learn Linear Regression (SGD) Section ---
print("\n--- Training Scikit-Learn Linear Regression (SGD, with Car_Name OHE) ---")
model_sgd = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=50)

# If a DataConversionWarning appears for y_train, using y_train.ravel() is the fix.
# model_sgd.fit(x_train, y_train.ravel()) # would be the change
model_sgd.fit(x_train, y_train)


predictions_sgd_test = model_sgd.predict(x_test)
mse_sgd_test = mean_squared_error(y_test, predictions_sgd_test)
print(f"Sklearn SGD Regressor - Test MSE: {mse_sgd_test:.4f}")

predictions_sgd_train = model_sgd.predict(x_train)
predictions_sgd_val = model_sgd.predict(x_val)

fig_sgd, axes_sgd = plt.subplots(1, 3, figsize=(18, 5))
fig_sgd.suptitle("Scikit-Learn SGD Regressor (Car_Name OHE): Actual vs. Predicted", fontsize=16)

axes_sgd[0].scatter(x=y_train, y=predictions_sgd_train, alpha=0.6)
axes_sgd[0].set_xlabel("Actual Present Price", fontsize=10)
axes_sgd[0].set_ylabel("Predicted Present Price",  fontsize=10)
axes_sgd[0].set_title("Training Set")
min_max_train_sgd = [min(y_train.min(), predictions_sgd_train.min()), max(y_train.max(), predictions_sgd_train.max())]
axes_sgd[0].plot(min_max_train_sgd, min_max_train_sgd, color='red', linestyle='--')

axes_sgd[1].scatter(x=y_test, y=predictions_sgd_test, alpha=0.6)
axes_sgd[1].set_xlabel("Actual Present Price", fontsize=10)
axes_sgd[1].set_ylabel("Predicted Present Price",  fontsize=10)
axes_sgd[1].set_title("Testing Set")
min_max_test_sgd = [min(y_test.min(), predictions_sgd_test.min()), max(y_test.max(), predictions_sgd_test.max())]
axes_sgd[1].plot(min_max_test_sgd, min_max_test_sgd, color='red', linestyle='--')

axes_sgd[2].scatter(x=y_val, y=predictions_sgd_val, alpha=0.6)
axes_sgd[2].set_xlabel("Actual Present Price", fontsize=10)
axes_sgd[2].set_ylabel("Predicted Present Price",  fontsize=10)
axes_sgd[2].set_title("Validation Set")
min_max_val_sgd = [min(y_val.min(), predictions_sgd_val.min()), max(y_val.max(), predictions_sgd_val.max())]
axes_sgd[2].plot(min_max_val_sgd, min_max_val_sgd, color='red', linestyle='--')

fig_sgd.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n--- MSE Comparison on Test Set (with Car_Name OHE) ---")
print(f"TensorFlow NN MSE: {mse_tf_test:.4f}")
print(f"Sklearn Linear Regression (Least Squares) MSE: {mse_lr_test:.4f}")
print(f"Sklearn SGD Regressor MSE: {mse_sgd_test:.4f}")
