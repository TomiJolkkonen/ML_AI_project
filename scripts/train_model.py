import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/sales_data.csv')
df = df[df['Store_Open'] == 1]  # Only use data from open months

# Features and label
X = df[['Marketing_Spend']]
y = df['Sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# Save model
model.save('models/sales_model.keras')

# Evaluate
loss = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss}")
