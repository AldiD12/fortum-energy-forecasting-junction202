# FORTUM ENERGY FORECASTING - JUNCTION 2025 HACKATHON
## Context for Vulnerability Analysis

### PROBLEM STATEMENT
- **Challenge**: Predict energy consumption for 112 customer groups in Finland
- **Data**: 4 years of hourly consumption data (2021-2024) + weather + prices
- **Output**: 48-hour hourly predictions + 12-month monthly predictions
- **Metric**: MAPE (Mean Absolute Percentage Error) - lower is better
- **Competition Level**: Junction 2025 hackathon (high-stakes, top-tier competition)

### CURRENT PERFORMANCE
- **Private customers**: 3.87% MAPE
- **Enterprise customers**: 5.08% MAPE  
- **Overall reported**: 8.11% MAPE (suspicious - may be calculation error)
- **Target for winning**: 2-3% MAPE

### DATA SOURCES AVAILABLE
1. **training_consumption.xlsx** - Wide format with 112 customer group columns
2. **groups.csv** - Customer metadata (region, type, contract, consumption level)
3. **training_prices.csv** - Electricity market prices (EUR/MWh)
4. **Weather API** - Historical + forecast weather data
5. **Test files** - 48 hourly + 12 monthly prediction templates

### CURRENT ARCHITECTURE

#### Data Processing Pipeline:
1. Load Excel sheets (consumption, groups, prices)
2. Fetch weather data from Open-Meteo API
3. Merge all data sources on timestamp
4. Feature engineering (time, weather, price features)
5. Reshape wide→long format (32K hours × 112 groups = 3.6M rows)
6. Add group metadata and lag features
7. Train separate models per customer type
8. Generate predictions with group-specific models

#### Model Strategy:
- **Approach**: Customer-type specific LightGBM models
- **Private model**: Trained on residential customers
- **Enterprise model**: Trained on business customers
- **Features**: 25 features including lags, weather, prices, time components

#### Feature Engineering:
```python
# Time features
hour, dayofweek, month, dayofyear
hour_sin, hour_cos, month_sin, month_cos
is_holiday, is_weekend

# Weather features  
temperature, humidity, precipitation, windspeed
is_cold, heating_degree
temp_hour_interaction, temp_weekend_interaction

# Price features (RECENTLY ADDED)
eur_per_mwh, price_ma_24, price_volatility, price_trend
price_hour_interaction, price_weekend_interaction

# Lag features
lag_1, lag_24, lag_168 (1h, 1day, 1week)
roll_24, roll_168 (rolling averages)

# Group features
customer_type, contract_type, consumption_level (one-hot encoded)
```

### KNOWN WEAKNESSES & VULNERABILITIES

#### 1. MODEL ARCHITECTURE ISSUES
- **Single algorithm**: Only LightGBM, no ensembling
- **Limited customer segmentation**: Only 2 models (Private/Enterprise)
- **No cross-group interactions**: Groups treated independently
- **Static hyperparameters**: No optimization performed

#### 2. FEATURE ENGINEERING GAPS
- **Insufficient lags**: Only 3 lag periods vs winning solutions use 10-15
- **No seasonal decomposition**: Missing trend/seasonal components
- **Limited weather interactions**: Only basic temperature interactions
- **No price elasticity modeling**: Price impact on demand not captured
- **Missing cross-correlations**: Industrial→residential demand relationships

#### 3. PREDICTION STRATEGY FLAWS
- **Static lag filling**: Uses historical values without autoregressive updates
- **No uncertainty quantification**: No confidence intervals
- **Simple monthly aggregation**: Naive scaling from hourly to monthly
- **No prediction smoothing**: Abrupt changes between hours

#### 4. VALIDATION WEAKNESSES
- **Simple time split**: 85/15 split, no cross-validation
- **No group-wise validation**: Some groups may be overfitting
- **No temporal validation**: Should validate on recent periods
- **Inconsistent MAPE calculation**: Overall MAPE doesn't match individual

#### 5. DATA UTILIZATION ISSUES
- **Underused price data**: Recently added but limited features
- **Single weather location**: All groups use Helsinki weather
- **No external data**: Missing holidays, economic indicators
- **Limited group clustering**: Basic customer_type segmentation only

### TECHNICAL IMPLEMENTATION DETAILS

#### Current Model Parameters:
```python
params = {
    'objective': 'regression',
    'metric': 'mape', 
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

#### Data Shapes:
- Training data: (32,856 hours, 113 columns)
- Long format: (3,679,872 rows, 46 features)
- Groups metadata: (112 groups, 7 attributes)
- Feature matrix: 25 final features after processing

#### Prediction Logic:
```python
# For each customer group:
1. Identify customer type (Private/Enterprise)
2. Select appropriate model
3. Get last 200 consumption values for lags
4. Create future feature matrix (48 hours)
5. Fill lags with static historical values
6. Predict using group-specific model
7. Scale hourly→monthly with simple averaging
```

### COMPETITIVE LANDSCAPE INSIGHTS

#### Winning Strategies (Based on Analysis):
1. **Ensemble modeling**: Multiple algorithms + stacking
2. **Advanced feature engineering**: 50+ features with interactions
3. **Autoregressive prediction**: Dynamic lag updating
4. **Proper validation**: Time series cross-validation
5. **Hyperparameter optimization**: Bayesian optimization
6. **Cross-group modeling**: Network effects between groups

#### Performance Benchmarks:
- **Current solution**: 3.87-5.08% MAPE per group
- **Competitive threshold**: 3-4% MAPE
- **Winning level**: 2-3% MAPE
- **Gap to close**: 1-2% improvement needed

### SPECIFIC VULNERABILITIES TO ANALYZE

#### High-Priority Issues:
1. **MAPE calculation inconsistency** - Why 8.11% overall vs 3-5% individual?
2. **Lag feature limitations** - Only 3 lags vs industry standard 10-15
3. **Missing autoregressive prediction** - Static vs dynamic lag updates
4. **Insufficient price modeling** - Basic features vs elasticity modeling
5. **No ensemble approach** - Single model vs multiple model combination

#### Medium-Priority Issues:
1. **Limited customer segmentation** - 2 groups vs optimal clustering
2. **Basic seasonal handling** - Simple cyclical vs decomposition
3. **Weather oversimplification** - Single location vs regional variations
4. **Validation methodology** - Simple split vs proper time series CV
5. **Feature selection** - No systematic feature importance analysis

#### Low-Priority Issues:
1. **Hyperparameter tuning** - Manual vs automated optimization
2. **Uncertainty quantification** - Point estimates vs confidence intervals
3. **External data integration** - Limited vs comprehensive data sources
4. **Prediction smoothing** - Raw predictions vs smoothed outputs
5. **Model interpretability** - Black box vs explainable predictions

### QUESTIONS FOR VULNERABILITY ANALYSIS

1. **What are the top 3 most critical weaknesses that could cost us the competition?**
2. **Which missing features would provide the biggest MAPE improvement?**
3. **Is the current model architecture fundamentally flawed for this problem?**
4. **What advanced techniques are we missing that competitors likely use?**
5. **How can we achieve 2-3% MAPE with minimal code changes?**
6. **What's causing the MAPE calculation inconsistency?**
7. **Should we completely redesign the approach or incrementally improve?**
8. **What are the biggest risks in our current prediction strategy?**
9. **Which validation approach would give us the most reliable performance estimate?**
10. **What ensemble strategies would work best for this time series problem?**

### SUCCESS CRITERIA
- **Primary**: Achieve <3% MAPE consistently across customer groups
- **Secondary**: Robust predictions that generalize to unseen data
- **Tertiary**: Interpretable model for business insights
- **Timeline**: Hackathon environment (limited time for major changes)

---

**ANALYSIS REQUEST**: Please identify the most critical vulnerabilities in this energy forecasting solution that could prevent it from winning the hackathon. Focus on actionable improvements that could yield 1-2% MAPE reduction with reasonable implementation effort.
#
## CURRENT IMPLEMENTATION CODE

```python
"""
FORTUM JUNCTION 2025 - Multi-Group Energy Forecasting
Forecasts consumption for 112 customer groups
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
import holidays
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'train_file': '20251111_JUNCTION_training.xlsx',
    'sheet_consumption': 'training_consumption',
    'sheet_groups': 'groups',
    'output_hourly': 'submission_hourly.csv',
    'output_monthly': 'submission_monthly.csv',
    'use_weather': True,
}

# 112 customer group IDs
CUSTOMER_GROUPS = [28,29,30,36,37,38,39,40,41,42,43,73,74,76,116,149,150,151,152,157,
                   196,197,198,199,200,201,213,222,225,231,233,234,235,237,238,270,271,
                   295,298,301,302,303,304,305,307,308,346,347,348,378,380,385,387,390,
                   391,393,394,395,396,397,398,399,400,401,402,403,404,405,447,450,459,
                   460,466,468,469,538,541,542,561,570,573,577,580,581,582,583,585,586,
                   622,623,624,625,626,657,658,659,682,691,692,693,694,695,697,698,705,
                   706,707,708,709,738,740,741]

# Key functions and model training logic...
# [Full implementation included in separate code block]
```

### SAMPLE OUTPUT DATA

#### Current Hourly Predictions (first few rows):
```
measured_at;28;29;30;36;37;38;39;40;41;42;43;...
2024-10-01T00:00:00.000Z;4,597993186236842;0,10955811342596312;0,5979019114611579;...
2024-10-01T01:00:00.000Z;4,626805799644689;0,11004114795859866;0,6034638492401875;...
```

#### Performance Metrics:
- Private customers: 3.87% MAPE
- Enterprise customers: 5.08% MAPE  
- Overall reported: 8.11% MAPE (inconsistent!)

### CRITICAL ANALYSIS POINTS

#### 1. **MAPE Calculation Bug** (HIGH PRIORITY)
The individual customer type MAPEs (3.87%, 5.08%) don't align with overall MAPE (8.11%). This suggests:
- Incorrect aggregation method
- Different validation sets being used
- Potential data leakage or overfitting

#### 2. **Static Lag Features** (HIGH PRIORITY)
```python
# Current approach - FLAWED:
group_future['lag_1'] = last_vals['consumption'].iloc[-1]  # Static!
group_future['lag_24'] = last_vals['consumption'].iloc[-24]  # Static!

# Should be autoregressive:
# lag_1 = previous_prediction
# lag_24 = prediction_from_24_hours_ago
```

#### 3. **Limited Feature Engineering** (MEDIUM PRIORITY)
Missing critical features that winning solutions typically have:
- Cross-group correlations (industrial affects residential)
- Seasonal decomposition (trend + seasonal + residual)
- Price elasticity modeling (demand response to price changes)
- Advanced weather interactions (heating/cooling degree days)

#### 4. **Naive Monthly Prediction** (MEDIUM PRIORITY)
```python
# Current approach - TOO SIMPLE:
base = monthly_avg[col] * 730  # Just scale hourly average
monthly_preds[col] = [base * (0.9 + np.random.rand() * 0.2) for _ in range(12)]
```

#### 5. **No Ensemble Strategy** (MEDIUM PRIORITY)
Single algorithm (LightGBM) vs winning approaches that typically use:
- Multiple algorithms (XGBoost, CatBoost, Neural Networks)
- Model stacking/blending
- Uncertainty quantification

### IMMEDIATE ACTION ITEMS

1. **Fix MAPE calculation** - Investigate why overall != individual
2. **Implement autoregressive prediction** - Update lags dynamically  
3. **Add more lag features** - Extend from 3 to 10-15 lags
4. **Enhance price features** - Add elasticity and interaction terms
5. **Improve monthly prediction** - Use proper seasonal patterns

### COMPETITIVE INTELLIGENCE

Based on hackathon best practices, winning solutions likely include:
- **Ensemble of 3-5 different algorithms**
- **50+ engineered features** with cross-group interactions
- **Proper time series validation** with walk-forward analysis
- **Hyperparameter optimization** using Bayesian methods
- **Autoregressive prediction** with dynamic lag updates
- **Uncertainty quantification** with confidence intervals

### RISK ASSESSMENT

**High Risk**: MAPE calculation inconsistency could indicate fundamental modeling errors
**Medium Risk**: Static lag approach will perform poorly on dynamic consumption patterns  
**Low Risk**: Missing advanced features (can be added incrementally)

---

**FINAL REQUEST**: Please provide a prioritized list of the top 5 most critical vulnerabilities that need immediate attention to achieve hackathon-winning performance (target: 2-3% MAPE).