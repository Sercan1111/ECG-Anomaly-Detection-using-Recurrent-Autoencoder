# ECG Anomaly Detection using Recurrent Autoencoder

## Overview
This project is dedicated to the detection of anomalies in electrocardiogram (ECG) signals using a Recurrent Autoencoder architecture. The primary goal is to identify unusual patterns in ECG signals that may indicate cardiac abnormalities. This Python-based project utilizes PyTorch for building and training the deep learning model, and it includes visualization of ECG signals using Matplotlib and Seaborn.

## Dataset
The dataset used in this project is based on ECG200, which is part of the UCR Time Series Classification Archive. It consists of labeled ECG signals where each category represents normal and abnormal rhythms.

## Preprocessing Steps
- **Data Loading**: Load ECG data from ARFF files.
- **Normalization**: Normalize ECG signal values to a standard scale.
- **Train-Test Split**: Split the dataset into training, validation, and testing sets to ensure model robustness.

## Model Architecture
- **Encoder**: Captures the important features of the ECG signal, reducing its dimensionality.
- **Decoder**: Attempts to reconstruct the ECG signal from the encoded representation.
- **Recurrent Neural Networks (LSTM)**: Used in both the encoder and decoder to handle the sequential nature of ECG signals.

## Training the Model
- The model is trained to minimize the reconstruction error, distinguishing between normal and anomalous signals based on the loss magnitude.
- Training involves multiple epochs with validation checks to prevent overfitting.

## Files in the Repository
- `train.py`: Main script for training the model.
- `model.py`: Contains the PyTorch model definitions.
- `utils.py`: Helper functions for data loading and preprocessing.
- `requirements.txt`: All necessary Python dependencies.

## How to Run
To run this project, follow these steps:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Train the model:
  python train.py
3. Evaluate the model by running the script with test data (this could be set up in train.py or a separate evaluate.py script).

## Results
- The model's performance can be evaluated using loss distribution plots to understand how well the model is identifying anomalies.
- Precision, Recall, and F1-Score metrics are calculated to quantify model performance.

## Future Work
- Implement more complex LSTM architectures or explore GRU (Gated Recurrent Unit) models.
- Integrate more datasets to improve the model's generalizability.
- Apply transfer learning from models trained on larger datasets.
## Contributions 
- Contributions to this project are welcome. Please fork the repository and open a pull request with proposed changes.
