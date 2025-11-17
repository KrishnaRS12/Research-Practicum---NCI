# Research-Practicum---NCI
## SpatioTemporal_CrimeForecast
Forecast daily theft counts for Chicago Community Area 28 using deep learning with exogenous weather & calendar signals. The project compares LSTM-family baselines with attention-augmented architectures and integrates LIME for model explainability.

Scope: The work intentionally focuses on one community area (Area 28) and one crime type (THEFT) to keep the spatial distribution consistent, reduce heterogeneity, and clearly analyze temporal patterns and model behavior.

### Features
  - Data pipeline:
    
        1. Daily aggregation of crimes with lags (t−1, t−7)
        2. Rolling weekday/month/weekend averages
        3. Weekend & US holiday indicators
        4. Merge with daily weather (temperature, humidity, wind speed) and Discomfort Index (DI)
    
  - Models:
    
        1. LSTM, BiLSTM, BiGRU
        2. BiLSTM + Attention (temporal attention)
        3. Spatio-Temporal CBAM Attention
    
  - Evaluation:
    
        1. MSE, RMSE, MAE, R²
        2. Inference latency & throughput, memory usage, execution time
    
  - Explainability:
    
        1. LIME for local feature importance (verifies role of recent counts, lags, weekday patterns)

### Setup
  - Environment:

        python -m venv .venv
        source .venv/bin/activate   # on Windows: .venv\Scripts\activate
        pip install --upgrade pip
        pip install -r requirements.txt


  - Data Access:
        - Due to file size limits, the datasets are hosted externally.

      1. Crime dataset (Chicago Open Data Portal): https://drive.google.com/file/d/1KZzHzDKm8kTE5amHTQDOmHRVMD3rN_qW/view?usp=drive_link
      2. Weather data (NOAA/Chicago): https://docs.google.com/spreadsheets/d/15jJ3_ix0WoLxUaaCUEXZvB8Gi0Yiv98I/edit?usp=drive_link&ouid=118004913183445890400&rtpof=true&sd=true
         


###  Results Summary

| Model                     | MSE     | RMSE   | MAE    | Latency (ms/pred) | Throughput (pred/s) | Test Prediction Time (s) | 7-Day Forecast Time (s) | Memory (MB) |
|---------------------------|---------|--------|--------|---------------------|-----------------------|----------------------------|---------------------------|-------------|
| LSTM                      | 0.7477  | 0.8647 | 0.6959 | 112.89              | 2801                  | 0.588                      | 0.7179                    | 1946        |
| BiLSTM                    | 0.7626  | 0.8733 | 0.6850 | 102.37              | 1839                  | 1.481                      | 0.6626                    | 2033        |
| BiGRU                     | 0.7279  | 0.8531 | 0.6869 | 106.93              | 1239                  | 1.617                      | 0.9254                    | 2099        |
| BiLSTM + Attention        | 0.7046  | 0.8394 | 0.6718 | 98.75               | 1378                  | 1.508                      | 0.8552                    | 2194        |
| BiLSTM + CBAM (ST-Attn)   | 0.7036  | 0.8388 | 0.6685 | 101.39              | 711                   | 2.831                      | 0.6082                    | 2079        |






> Note: Metrics are based on an 80:20 time-based split.  
> Models were trained on the same feature-engineered dataset for comparability.


### Notes

      - The code uses TensorFlow 2.17, Keras, and LIME for model training and interpretation.
      - All preprocessing steps, feature engineering, and model evaluations are included in the Jupyter notebook.
      - ARIMA and SARIMAX were tested separately but excluded due to poor performance (constant future values).
