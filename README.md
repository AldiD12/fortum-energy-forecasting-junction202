# 🔋 Fortum Energy Forecasting - Junction 2025

Multi-customer group energy consumption forecasting solution for the Junction 2025 hackathon.

## 🎯 Challenge Overview

Predict energy consumption for **112 customer groups** in Finland using:
- 4 years of hourly consumption data (2021-2024)
- Weather data from Open-Meteo API
- Electricity market prices
- Customer group metadata

## 📊 Current Performance

- **Private customers**: 3.87% MAPE
- **Enterprise customers**: 5.08% MAPE
- **Target for winning**: 2-3% MAPE

## 🏗️ Architecture

### Data Pipeline
1. Load Excel sheets (consumption, groups, prices)
2. Fetch weather data from Open-Meteo API
3. Feature engineering (time, weather, price interactions)
4. Reshape to long format (3.6M rows)
5. Train customer-type specific LightGBM models
6. Generate 48-hour + 12-month predictions

### Key Features
- **Time features**: Hour, day, month with cyclical encoding
- **Weather features**: Temperature, humidity, precipitation, wind
- **Price features**: Market prices with moving averages and volatility
- **Lag features**: 1h, 24h, 168h consumption lags
- **Group features**: Customer type, contract type, consumption level

### Model Strategy
- **Customer-type specific models**: Separate LightGBM for Private/Enterprise
- **Advanced parameters**: Optimized for competition performance
- **Group-aware prediction**: Uses appropriate model per customer group

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy lightgbm scikit-learn requests holidays openpyxl
```

### Run the Model
```bash
python app.py
```

### Expected Outputs
- `submission_hourly.csv` - 48 hourly predictions for all 112 groups
- `submission_monthly.csv` - 12 monthly predictions for all groups
- `fortum_models.pkl` - Trained model artifacts

## 📁 Project Structure

```
enerlytics/
├── app.py                              # Main forecasting pipeline
├── requirements.txt                    # Python dependencies
├── hackathon_context_for_analysis.md   # Vulnerability analysis context
├── winning_improvements.py             # Advanced strategies for winning
├── submission_hourly.csv               # Hourly predictions output
├── submission_monthly.csv              # Monthly predictions output
└── README.md                          # This file
```

## 🔧 Key Improvements Implemented

### ✅ Recently Added
- **Electricity price features** (critical for demand forecasting)
- **Customer-type specific models** (Private vs Enterprise)
- **Advanced weather interactions** (temperature × hour, temperature × weekend)
- **Price interaction features** (price × hour, price × weekend)

### 🎯 Winning Strategies Identified
- **Autoregressive prediction** (dynamic lag updates)
- **Cross-group correlations** (industrial affects residential)
- **Ensemble modeling** (multiple algorithms)
- **Advanced feature engineering** (50+ features)
- **Proper time series validation**

## 📈 Performance Analysis

### Strengths
- Group-specific modeling approach
- Comprehensive feature engineering
- Real-time weather integration
- Competitive individual group performance

### Areas for Improvement
- MAPE calculation inconsistency
- Static lag feature approach
- Limited ensemble strategies
- Basic monthly prediction method

## 🏆 Competition Strategy

### Immediate Wins (1-2% MAPE improvement)
1. Fix autoregressive prediction with dynamic lags
2. Add cross-group interaction features
3. Implement proper seasonal decomposition
4. Enhance price elasticity modeling

### Advanced Techniques
1. Model stacking/ensembling
2. Hyperparameter optimization with Optuna
3. Uncertainty quantification
4. Feature selection with SHAP

## 📊 Data Sources

- **Training Data**: `20251111_JUNCTION_training.xlsx`
  - `training_consumption`: Wide format consumption data
  - `groups`: Customer group metadata
  - `training_prices`: Electricity market prices
- **Weather**: Open-Meteo API (historical + forecast)
- **Test Templates**: Hourly and monthly prediction formats

## 🤝 Contributing

This is a hackathon project. Key areas for contribution:
- Advanced feature engineering
- Model ensembling strategies
- Validation methodology improvements
- Performance optimization

## 📄 License

MIT License - Feel free to use and modify for your own energy forecasting projects.

---

**Junction 2025 Hackathon** | **Team**: Energy Analytics | **Target**: Top 3 Finish