import jax
import jax.numpy as jnp
import pandas as pd
from compute import Compute
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("synthetic_car_data.csv")

    features = list(df.columns)
    print(f"available columns: {features}")

    data = jnp.array(df.values)
    X, y = data[:, :-1], data[:, -1:]
    X, y = X.astype(jnp.float32), y.astype(jnp.float32)

    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Split train/test
    index = int(0.8 * num_samples)
    X_train, X_test = X[:index], X[index:]
    y_train, y_test = y[:index], y[index:]

    # Normalize (using training statistics)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std  

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    cmp = Compute(num_features)
    cmp.gradient_descent(X_train, y_train,iter=1000)


    #test it 
    y_pred = cmp.lr_model(X_test)

    test_loss = cmp.MSE_loss(cmp.w, cmp.b, X_test, y_test)
    print(f"\nTest MSE (normalized y): {test_loss:.4f}")

    #denormalize predictions to original scale
    y_pred_orig = y_pred * y_std + y_mean
    y_test_orig = y_test * y_std + y_mean
    mse_orig = jnp.mean((y_pred_orig - y_test_orig)**2)
    print(f"Test MSE (original scale): {mse_orig:.4f}") #still wrong 

    plt.figure(figsize=(7,5))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs True (Test Set)")
    plt.plot([y_test_orig.min(), y_test_orig.max()],
            [y_test_orig.min(), y_test_orig.max()], 'r--')  #perfect prediction line
    plt.show()

