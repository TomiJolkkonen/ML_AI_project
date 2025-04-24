# Sales Prediction with Machine Learning

This project demonstrates a simple machine learning pipeline using TensorFlow and Keras to predict monthly sales from marketing spending.

## 🧠 Objective

Predict `Sales` based on `Marketing_Spend` using a neural network.

## 🗂️ Project Structure

ml_sales_prediction_project/ ├── data/ # Sample CSV data ├── models/ # Trained ML model ├── notebooks/ # For Jupyter Notebooks (optional) ├── scripts/ # Python script for training └── README.md # Project instructions

## 🛠️ Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install tensorflow pandas scikit-learn

  How to run

  python scripts/train_model.py

  Output

  prints test loss (MSE)
  saves trained model for reuse