# Cryptocurrency-Price-Analysis-Using-DL
Introduction
This project focuses on predicting the future prices of popular cryptocurrencies such as Bitcoin (BTC), Ethereum (ETH), and Ripple (XRP) using deep learning models. Given the volatile and non-linear nature of cryptocurrency markets, this project implements Long Short-Term Memory (LSTM) networks and Artificial Neural Networks (ANN) to model and forecast price trends effectively.

Objective
The primary objective of this project is to design and evaluate deep learning models capable of forecasting future cryptocurrency prices based on historical data. The models aim to minimize prediction error while effectively capturing the time-dependent patterns in crypto market data.

Dataset
Historical cryptocurrency data was collected from Yahoo Finance using the yFinance library. The dataset includes daily records of Open, High, Low, Close, and Volume for Bitcoin (BTC-USD), Ethereum (ETH-USD), and Ripple (XRP-USD). The data spans several years to ensure the model has enough historical context to learn from.

Data Preprocessing
The raw data was cleaned and scaled using MinMaxScaler to ensure all features are on the same scale. Missing values were handled appropriately, and a windowing technique was applied to convert the data into a supervised learning format. For LSTM, the input was reshaped into a 3D structure to match its expected input shape, while for ANN, the input was flattened accordingly.

Model Architectures
LSTM (Long Short-Term Memory)
LSTM is a type of recurrent neural network (RNN) designed to handle sequence prediction problems. In this project, the LSTM model learns from time-series data by capturing temporal dependencies. The model consists of one or more LSTM layers followed by dense layers to output the predicted price.

ANN (Artificial Neural Network)
ANN is a feedforward neural network that maps input features to outputs through hidden layers of neurons. While it does not inherently handle temporal dependencies, it is included in this project as a comparative baseline. The ANN model is built with multiple fully connected layers and ReLU activation functions, with the final layer outputting the predicted price.

Training Procedure
Both models were trained using historical closing price data. The dataset was split into training and testing sets based on a time-based split. The models were trained using the Mean Squared Error (MSE) loss function and optimized with the Adam optimizer. Training was conducted over multiple epochs with appropriate batch sizes, and early stopping was applied to prevent overfitting.

Evaluation Metrics
To assess the performance of the models, the following metrics were used:

Root Mean Squared Error (RMSE): Quantifies the average magnitude of the prediction error.

Mean Absolute Percentage Error (MAPE): Measures the percentage error in predictions.

These metrics were calculated for each cryptocurrency and compared across both models.

Results
The LSTM model outperformed the ANN model in capturing temporal patterns and exhibited better forecasting accuracy across all three cryptocurrencies. ANN provided reasonable short-term predictions but lacked the ability to model sequential dependencies as effectively as LSTM. Both models showed improvements over simple statistical baselines.

Project Structure
The project is organized into the following directories and files:

data/ – Contains raw and processed datasets.

models/ – Includes saved model weights and architecture definitions.

notebooks/ – Jupyter notebooks for exploration and model development.

utils/ – Utility scripts for data preprocessing and visualization.

main.py – Main script to train and evaluate models.

requirements.txt – Python dependencies required to run the project.

How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/crypto-price-prediction-DL.git
cd crypto-price-prediction-DL
Install the required dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Run the main script to train and evaluate the models:

css
Copy
Edit
python main.py
