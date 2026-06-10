# A Decomposition-Based Sequential Deep Learning Model for Time Series Forecasting: SSD-GRU for Wind Data

This repository contains the MATLAB and Python code associated with the paper:

**A Decomposition-Based Sequential Deep Learning Model for Time Series Forecasting: SSD-GRU for Wind Data**

## Overview

The proposed SSD-GRU framework forecasts wind speed data using a two-stage process:

1. **SSD decomposition in MATLAB**  
   The original wind speed time series is decomposed into three components:
   - Trend component
   - Oscillatory component
   - Noise component

2. **GRU forecasting in Python TensorFlow/Keras**  
   Each decomposed component is trained and predicted separately using a GRU deep learning model.

3. **Forecast reconstruction**  
   The final wind speed forecast is reconstructed by combining the predicted components:

```text
Forecasted wind speed = predicted trend + predicted oscillatory component + predicted noise
