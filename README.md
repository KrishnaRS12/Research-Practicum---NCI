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
        3. BiLSTM + CBAM + Informer (sparse attention for long sequences + channel/spatial attention)
        4. Spatio-Temporal CBAM Attention
    
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
         
         
### Results Summary

| Model | RMSE | MAE | Latency | Throughput | Memory Usage | Remarks |
|:------|:----:|:----:|:--------:|:------------:|:--------------:|:---------|
| **LSTM** | 0.86 | 0.70 | High | Low | Moderate | Baseline sequential model |
| **BiLSTM** | 0.85 | 0.68 | Moderate | Moderate | Moderate | Better capture of bidirectional temporal patterns |
| **BiGRU** | 0.85 | 0.68 | **Low** | **High** | **Low** | Most efficient recurrent alternative |
| **BiLSTM + Attention** | **0.84** | **0.67** | Moderate | Moderate | Moderate | **Best-performing model overall** |
| **Informer + CBAM** | 0.85 | 0.68 | Moderate | High | High | Effective for long-sequence modeling |
| **Spatio-Temporal CBAM Attention** | 0.85 | **0.67** | Slightly higher | Moderate | High | Close second; strong MAE stability |



> Note: Metrics are based on an 80:20 time-based split.  
> Models were trained on the same feature-engineered dataset for comparability.


### Notes

      - The code uses TensorFlow 2.17, Keras, and LIME for model training and interpretation.
      - All preprocessing steps, feature engineering, and model evaluations are included in the Jupyter notebook.
      - ARIMA and SARIMAX were tested separately but excluded due to poor performance (constant future values).
