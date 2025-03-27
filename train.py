import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def predict(X,M):
    weights = M[0:-1]
    bias = M[-1]
    prediction = sum(X[i] * weights[i] for i in range(len(M)-1)) + bias
    return prediction

def compute_gradients(X, y, M):
    N = len(M)
    dm = [0] * N
    for i in range(len(X)):
        y_pred = predict(X[i], M)
        error = y[i] - y_pred
        for j in range(N-1):
            dm[j] += (-2/N) * error * X[i][j]
        dm[-1] += (-2/N)*error
    return dm

def update_weights(M, dm, a):
    for i in range(len(M)):
        M[i] -= a * dm[i]
    return M

def train(X, y, M, a, max_epochs = 10**5, tolerance = 1e-4, mse_threshold = 0.95):
    previous_mse = float("inf")
    for epoch in range(max_epochs):
        dm = compute_gradients(X, y, M)
        M = update_weights(M, dm, a)
        mse = sum((y[i] - predict(X[i], M))**2 for i in range(len(X))) / len(X)
        print(f"Epoch {epoch+1}/{max_epochs}: MSE = {mse}")
        if mse == float("inf") or mse > 1e10:
            print(f"Training stopped at epoch {epoch}: MSE too large ({mse})")
            break

        if mse < mse_threshold:
            break
        if abs(previous_mse - mse) < tolerance:
            print(f"Training stopped at epoch {epoch}, MSE: {mse:.4f} (Below threshold)")
            break
        previous_mse = mse
    return M


# Load dataset
file_path = os.getenv("DATASET_PATH")
df = pd.read_excel(file_path, engine="openpyxl")  
X = df.iloc[:, :-1].values  # Features (All columns except last)
y = df.iloc[:, -1].values  # Target (Last column)
X = (X - X.min()) / (X.max() - X.min())  # Normalize Features
# Initialize Weights and Hyperparameters
M = [0] * (10 + 1)  # Weights + Bias
alpha = 1e-6 # Learning Rate
epochs = 10**5  # Maximum Training Steps

# Train the Model
M_final = train(X, y, M, alpha, epochs)

# Save Trained Weights
pd.DataFrame(M_final).to_csv("model.csv", index=False, header=False)
print("Training completed. Weights saved to model.csv.")
