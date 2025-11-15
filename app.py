"""
FORTUM JUNCTION 2025 - Multi-Group Energy Forecasting
Forecasts consumption for 112 customer groups
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
    'combined_data': 'combined_data.csv',  # Combined weather + consumption + price data
    'train_file': '20251111_JUNCTION_training.xlsx',  # Still need for groups metadata
    'sheet_groups': 'groups',
    'output_hourly': 'submission_hourly.csv',
    'output_monthly': 'submission_monthly.csv',
}

# 112 customer group IDs
CUSTOMER_GROUPS = [28,29,30,36,37,38,39,40,41,42,43,73,74,76,116,149,150,151,152,157,
                   196,197,198,199,200,201,213,222,225,231,233,234,235,237,238,270,271,
                   295,298,301,302,303,304,305,307,308,346,347,348,378,380,385,387,390,
                   391,393,394,395,396,397,398,399,400,401,402,403,404,405,447,450,459,
                   460,466,468,469,538,541,542,561,570,573,577,580,581,582,583,585,586,
                   622,623,624,625,626,657,658,659,682,691,692,693,694,695,697,698,705,
                   706,707,708,709,738,740,741]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title.center(70)}")
    print(f"{'='*70}\n")

def get_weather(start_date, end_date):
    """Fetch historical weather from Open-Meteo"""
    print(f"Fetching weather: {start_date.date()} to {end_date.date()}")
    
    all_weather = []
    current = start_date
    
    while current < end_date:
        chunk_end = min(current + timedelta(days=365), end_date)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 60.1699,
            "longitude": 24.9384,
            "start_date": current.strftime('%Y-%m-%d'),
            "end_date": chunk_end.strftime('%Y-%m-%d'),
            "hourly": ["temperature_2m", "relativehumidity_2m", "precipitation", "windspeed_10m"],
            "timezone": "Europe/Helsinki"
        }
        
        try:
            response = requests.get(url, params=params, timeout=90)
            data = response.json()
            
            chunk_df = pd.DataFrame({
                'measured_at': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relativehumidity_2m'],
                'precipitation': data['hourly']['precipitation'],
                'windspeed': data['hourly']['windspeed_10m']
            })
            all_weather.append(chunk_df)
            print(f"  ✅ {current.year}: {len(chunk_df)} records")
        except Exception as e:
            print(f"  ⚠️ {current.year}: Using fallback")
            dates = pd.date_range(current, chunk_end, freq='H')
            all_weather.append(pd.DataFrame({
                'measured_at': dates,
                'temperature': 5, 'humidity': 70, 'precipitation': 0, 'windspeed': 5
            }))
        
        current = chunk_end + timedelta(days=1)
    
    return pd.concat(all_weather, ignore_index=True)

def get_forecast(days=7):
    """Get weather forecast"""
    print(f"Fetching {days}-day forecast...")
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 60.1699,
        "longitude": 24.9384,
        "hourly": ["temperature_2m", "relativehumidity_2m", "precipitation", "windspeed_10m"],
        "forecast_days": min(days, 16),
        "timezone": "Europe/Helsinki"
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        data = response.json()
        
        forecast = pd.DataFrame({
            'measured_at': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relativehumidity_2m'],
            'precipitation': data['hourly']['precipitation'],
            'windspeed': data['hourly']['windspeed_10m']
        })
        print(f"  ✅ {len(forecast)} records")
        return forecast
    except:
        print(f"  ⚠️ Using average weather")
        return None

def create_features(df):
    """Create time and enhanced weather features"""
    df = df.copy()

    # Time features
    df['hour'] = df['measured_at'].dt.hour
    df['dayofweek'] = df['measured_at'].dt.dayofweek
    df['month'] = df['measured_at'].dt.month
    df['dayofyear'] = df['measured_at'].dt.dayofyear

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Calendar
    years = df['measured_at'].dt.year.unique()
    fi_holidays = holidays.Finland(years=years)
    df['is_holiday'] = df['measured_at'].dt.date.apply(lambda x: x in fi_holidays).astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # ENHANCED WEATHER FEATURES (from combined_data.csv)
    if 'avg_temp' in df.columns:
        # Temperature range (daily variation)
        if 'max_temp' in df.columns and 'min_temp' in df.columns:
            df['temp_range'] = df['max_temp'] - df['min_temp']
            df['temp_range_high'] = (df['temp_range'] > 10).astype(int)  # High variation day

        # Heating/cooling needs
        df['is_cold'] = (df['avg_temp'] < -5).astype(int)
        df['is_hot'] = (df['avg_temp'] > 25).astype(int)
        df['heating_degree'] = np.maximum(17 - df['avg_temp'], 0)
        df['cooling_degree'] = np.maximum(df['avg_temp'] - 24, 0)

        # Temperature interactions
        df['temp_hour_interaction'] = df['avg_temp'] * df['hour']
        df['temp_weekend_interaction'] = df['avg_temp'] * df['is_weekend']

    # Wind features
    if 'wind_speed' in df.columns:
        df['is_windy'] = (df['wind_speed'] > 8).astype(int)
        df['wind_chill_effect'] = df['wind_speed'] * (df['avg_temp'] if 'avg_temp' in df.columns else 0)

        # Wind direction categories (N, E, S, W)
        if 'wind_direction' in df.columns:
            df['wind_north'] = ((df['wind_direction'] >= 315) | (df['wind_direction'] < 45)).astype(int)
            df['wind_east'] = ((df['wind_direction'] >= 45) & (df['wind_direction'] < 135)).astype(int)
            df['wind_south'] = ((df['wind_direction'] >= 135) & (df['wind_direction'] < 225)).astype(int)
            df['wind_west'] = ((df['wind_direction'] >= 225) & (df['wind_direction'] < 315)).astype(int)

    # Air pressure (weather stability indicator)
    if 'air_pressure' in df.columns:
        df['pressure_high'] = (df['air_pressure'] > 1020).astype(int)  # High pressure = stable weather
        df['pressure_low'] = (df['air_pressure'] < 1000).astype(int)   # Low pressure = stormy
        df['pressure_change'] = df['air_pressure'].diff().fillna(0)     # Rapid changes affect consumption

    # Precipitation
    if 'precipitation' in df.columns:
        df['is_raining'] = (df['precipitation'] > 0.1).astype(int)
        df['heavy_rain'] = (df['precipitation'] > 5).astype(int)

    # Humidity
    if 'humidity' in df.columns:
        df['is_humid'] = (df['humidity'] > 80).astype(int)
        df['is_dry'] = (df['humidity'] < 40).astype(int)

    # Price interactions (WINNING FEATURES!)
    if 'eur_per_mwh' in df.columns:
        df['price_hour_interaction'] = df['eur_per_mwh'] * df['hour']
        df['price_weekend_interaction'] = df['eur_per_mwh'] * df['is_weekend']

        # Price-temperature interaction (expensive heating days)
        if 'avg_temp' in df.columns:
            df['price_temp_interaction'] = df['eur_per_mwh'] * df['heating_degree']

    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_section("FORTUM JUNCTION 2025")
    
    # ==================
    # 1. LOAD COMBINED DATA (Weather + Consumption + Prices)
    # ==================
    print_section("LOADING COMBINED DATA")

    # Load combined CSV with all data
    df = pd.read_csv(CONFIG['combined_data'])

    # Rename columns to standardized names
    column_mapping = {
        'timestamp': 'measured_at',
        'Average temperature [°C]': 'avg_temp',
        'Maximum temperature [°C]': 'max_temp',
        'Minimum temperature [°C]': 'min_temp',
        'Average relative humidity [%]': 'humidity',
        'Wind speed [m/s]': 'wind_speed',
        'Average wind direction [°]': 'wind_direction',
        'Precipitation [mm]': 'precipitation',
        'Average air pressure [hPa]': 'air_pressure',
    }
    df = df.rename(columns=column_mapping)

    # Parse timestamp
    df['measured_at'] = pd.to_datetime(df['measured_at'])
    df = df.sort_values('measured_at').reset_index(drop=True)

    print(f"✅ Combined data loaded: {df.shape}")
    print(f"   Date range: {df['measured_at'].min()} to {df['measured_at'].max()}")
    print(f"   Weather features: avg_temp, max_temp, min_temp, humidity, wind_speed, wind_direction, precipitation, air_pressure")
    print(f"   Price feature: eur_per_mwh")
    print(f"   Customer groups: {len([c for c in df.columns if c.isdigit()])}")

    # Load customer group metadata
    df_groups = pd.read_excel(CONFIG['train_file'], sheet_name=CONFIG['sheet_groups'])
    print(f"✅ Groups metadata: {df_groups.shape}")

    # ==================
    # 2. PRICE FEATURES (from combined data)
    # ==================
    print_section("PRICE FEATURES")

    df['price_ma_24'] = df['eur_per_mwh'].rolling(24).mean()
    df['price_volatility'] = df['eur_per_mwh'].rolling(24).std()
    df['price_trend'] = df['eur_per_mwh'].diff(24)
    df['price_ma_168'] = df['eur_per_mwh'].rolling(168).mean()  # Weekly average
    print("✅ Price features created (ma_24, volatility, trend, ma_168)")
    
    # ==================
    # 3. FEATURES
    # ==================
    print_section("FEATURE ENGINEERING")
    df = create_features(df)
    print(f"✅ Features created: {len(df.columns)} columns")
    
    # ==================
    # 4. CROSS-GROUP INTERACTION FEATURES
    # ==================
    print_section("CROSS-GROUP INTERACTIONS")

    # Calculate aggregate consumption by customer type
    private_groups = df_groups[df_groups['customer_type'] == 'Private']['region_id'].tolist()
    enterprise_groups = df_groups[df_groups['customer_type'] == 'Enterprise']['region_id'].tolist()

    private_cols = [str(g) for g in private_groups if g in df.columns]
    enterprise_cols = [str(g) for g in enterprise_groups if g in df.columns]

    if private_cols:
        df['total_private_consumption'] = df[private_cols].sum(axis=1)
        df['avg_private_consumption'] = df[private_cols].mean(axis=1)
        print(f"✅ Private group features: {len(private_cols)} groups")

    if enterprise_cols:
        df['total_enterprise_consumption'] = df[enterprise_cols].sum(axis=1)
        df['avg_enterprise_consumption'] = df[enterprise_cols].mean(axis=1)
        print(f"✅ Enterprise group features: {len(enterprise_cols)} groups")

    # Cross-group ratio (industrial affects residential)
    if private_cols and enterprise_cols:
        df['enterprise_to_private_ratio'] = df['total_enterprise_consumption'] / (df['total_private_consumption'] + 1)
        print("✅ Cross-group ratio feature added")

    # ==================
    # 5. RESHAPE TO LONG
    # ==================
    print_section("RESHAPING TO LONG FORMAT")

    # Get group columns that exist
    group_cols = [g for g in CUSTOMER_GROUPS if g in df.columns]
    print(f"Found {len(group_cols)} customer groups in data")

    id_vars = [c for c in df.columns if c not in group_cols]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=group_cols,
        var_name='group_id',
        value_name='consumption'
    )
    df_long['group_id'] = df_long['group_id'].astype(int)
    
    # Ensure group_id types match for merging
    df_groups['region_id'] = df_groups['region_id'].astype(int)
    
    # Merge group metadata
    df_long = df_long.merge(df_groups, left_on='group_id', right_on='region_id', how='left')
    
    print(f"✅ Long format: {df_long.shape}")
    
    # ==================
    # 5. GROUP FEATURES
    # ==================
    print("Adding group features...")
    
    # One-hot encode group characteristics
    for col in ['customer_type', 'contract_type', 'consumption_level']:
        if col in df_long.columns:
            dummies = pd.get_dummies(df_long[col], prefix=col, drop_first=True)
            df_long = pd.concat([df_long, dummies], axis=1)
    
    # Lag features per group - EXTENDED FOR BETTER PERFORMANCE
    print("Creating extended lags (10 lags + rolling windows)...")
    df_long = df_long.sort_values(['group_id', 'measured_at'])

    # Extended lag features: short-term, daily, and weekly patterns
    lag_periods = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]  # Up to 2 weeks
    for lag in lag_periods:
        df_long[f'lag_{lag}'] = df_long.groupby('group_id')['consumption'].shift(lag)

    # Rolling windows for trend detection
    for window in [24, 48, 168]:
        df_long[f'roll_{window}'] = df_long.groupby('group_id')['consumption'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    df_long = df_long.dropna(subset=['lag_1'])
    print(f"✅ After lags: {df_long.shape}")
    
    # ==================
    # 6. TRAIN MODEL
    # ==================
    print_section("TRAINING MODEL")
    
    # Features
    exclude = ['measured_at', 'consumption', 'group_id', 'region_id', 
               'region_name', 'subregion', 'macro_region',
               'customer_type', 'contract_type', 'consumption_level']
    
    feature_cols = [c for c in df_long.columns if c not in exclude and df_long[c].dtype in [np.float64, np.int64]]
    print(f"Features: {len(feature_cols)}")
    
    # Split - 90/10 for maximum training data
    split = int(len(df_long) * 0.90)
    X_train = df_long.iloc[:split][feature_cols]
    y_train = df_long.iloc[:split]['consumption']
    X_test = df_long.iloc[split:][feature_cols]
    y_test = df_long.iloc[split:]['consumption']

    print(f"Train (90%): {len(X_train):,} | Test (10%): {len(X_test):,}")
    
    # Train ENSEMBLE models for different customer types (WINNING STRATEGY!)
    models = {}

    # Store all predictions for proper MAPE calculation
    all_test_preds = []
    all_test_actuals = []

    for customer_type in df_groups['customer_type'].unique():
        print(f"\n{'='*50}")
        print(f"Training ENSEMBLE for {customer_type}...")
        print(f"{'='*50}")

        type_groups = df_groups[df_groups['customer_type'] == customer_type]['region_id'].tolist()
        type_data = df_long[df_long['group_id'].isin(type_groups)]

        if len(type_data) < 1000:
            continue

        # Split for this customer type - 90/10 for maximum training data
        type_split = int(len(type_data) * 0.90)
        X_type_train = type_data.iloc[:type_split][feature_cols]
        y_type_train = type_data.iloc[:type_split]['consumption']
        X_type_test = type_data.iloc[type_split:][feature_cols]
        y_type_test = type_data.iloc[type_split:]['consumption']

        # --- MODEL 1: LightGBM ---
        print(f"  Training LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'mape',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }

        lgb_train = lgb.Dataset(X_type_train, y_type_train)
        lgb_eval = lgb.Dataset(X_type_test, y_type_test, reference=lgb_train)

        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_eval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # --- MODEL 2: XGBoost ---
        print(f"  Training XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mape',
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0
        }

        xgb_train = xgb.DMatrix(X_type_train, label=y_type_train)
        xgb_eval = xgb.DMatrix(X_type_test, label=y_type_test)

        xgb_model = xgb.train(
            xgb_params, xgb_train, num_boost_round=500,
            evals=[(xgb_eval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Store both models as ensemble
        models[customer_type] = {
            'lgb': lgb_model,
            'xgb': xgb_model
        }

        # Calculate ensemble predictions (weighted average)
        lgb_pred = lgb_model.predict(X_type_test)
        xgb_pred = xgb_model.predict(xgb.DMatrix(X_type_test))

        # Ensemble: 60% LightGBM + 40% XGBoost (LightGBM typically performs better on this data)
        ensemble_pred = 0.6 * lgb_pred + 0.4 * xgb_pred

        # Calculate MAPE for ensemble
        lgb_mape = np.mean(np.abs((y_type_test - lgb_pred) / y_type_test)) * 100
        xgb_mape = np.mean(np.abs((y_type_test - xgb_pred) / y_type_test)) * 100
        ensemble_mape = np.mean(np.abs((y_type_test - ensemble_pred) / y_type_test)) * 100

        print(f"  LightGBM MAPE: {lgb_mape:.2f}%")
        print(f"  XGBoost MAPE: {xgb_mape:.2f}%")
        print(f"  ✅ Ensemble MAPE: {ensemble_mape:.2f}%")

        # Store ensemble predictions for overall MAPE calculation
        all_test_preds.extend(ensemble_pred.tolist())
        all_test_actuals.extend(y_type_test.tolist())

    # Calculate OVERALL MAPE correctly (weighted by all customer types)
    if all_test_preds:
        overall_mape = np.mean(np.abs((np.array(all_test_actuals) - np.array(all_test_preds)) / np.array(all_test_actuals))) * 100
        print(f"\n✅ Overall MAPE (all customer types): {overall_mape:.2f}%")
    else:
        overall_mape = 0.0
        print("\n⚠️  No predictions made")
    
    # ==================
    # 7. HOURLY FORECAST (48 hours)
    # ==================
    print_section("HOURLY PREDICTIONS")
    
    last_time = df['measured_at'].max()
    future_hours = pd.date_range(start=last_time + timedelta(hours=1), periods=48, freq='H')

    # Create future dataframe with weather forecasts/estimates
    future_df = pd.DataFrame({'measured_at': future_hours})

    # Use recent average weather as baseline forecast (simple but effective)
    recent_weather = df.tail(168)  # Last week
    for col in ['avg_temp', 'max_temp', 'min_temp', 'humidity', 'wind_speed',
                'wind_direction', 'precipitation', 'air_pressure', 'eur_per_mwh']:
        if col in df.columns:
            # Use hour-of-day average from last week for better forecasts
            hour_avgs = recent_weather.groupby(recent_weather['measured_at'].dt.hour)[col].mean()
            future_df[col] = future_df['measured_at'].dt.hour.map(hour_avgs)
            # Fill any missing with overall average
            future_df[col] = future_df[col].fillna(recent_weather[col].mean())

    print(f"✅ Future weather estimated from recent patterns (last 168 hours)")

    future_df = create_features(future_df)

    # Add cross-group features to future predictions (use last known values)
    if 'total_private_consumption' in df.columns:
        future_df['total_private_consumption'] = df['total_private_consumption'].iloc[-1]
        future_df['avg_private_consumption'] = df['avg_private_consumption'].iloc[-1]
    if 'total_enterprise_consumption' in df.columns:
        future_df['total_enterprise_consumption'] = df['total_enterprise_consumption'].iloc[-1]
        future_df['avg_enterprise_consumption'] = df['avg_enterprise_consumption'].iloc[-1]
    if 'enterprise_to_private_ratio' in df.columns:
        future_df['enterprise_to_private_ratio'] = df['enterprise_to_private_ratio'].iloc[-1]

    # Predict for each group using appropriate model with AUTOREGRESSIVE LAG UPDATING
    hourly_preds = {}

    for group_id in CUSTOMER_GROUPS:
        # Get customer type for this group
        group_info = df_groups[df_groups['region_id'] == group_id]
        if group_info.empty:
            hourly_preds[str(group_id)] = [1000] * 48
            continue

        customer_type = group_info['customer_type'].iloc[0]

        # Use appropriate ensemble models
        if customer_type in models:
            ensemble = models[customer_type]
            lgb_model = ensemble['lgb']
            xgb_model = ensemble['xgb']
        else:
            # Fallback to first available ensemble
            ensemble = list(models.values())[0]
            lgb_model = ensemble['lgb']
            xgb_model = ensemble['xgb']

        # Get last values for this group
        last_vals = df_long[df_long['group_id'] == group_id].tail(200)

        if len(last_vals) == 0:
            hourly_preds[str(group_id)] = [1000] * 48
            continue

        # AUTOREGRESSIVE PREDICTION - Update lags dynamically
        group_predictions = []
        history = last_vals['consumption'].tolist()  # Keep history for lag calculations

        for hour_idx in range(48):
            # Create features for THIS SPECIFIC HOUR
            hour_features = future_df.iloc[hour_idx:hour_idx+1].copy()

            # Add group features
            if not group_info.empty:
                for col in ['customer_type', 'contract_type', 'consumption_level']:
                    if col in group_info.columns:
                        val = group_info[col].values[0]
                        # Add dummy features
                        for cat in df_long[col].unique():
                            if cat != val:  # drop_first
                                hour_features[f'{col}_{cat}'] = 0
                            else:
                                if f'{col}_{cat}' in feature_cols:
                                    hour_features[f'{col}_{cat}'] = 1

            # DYNAMIC LAG UPDATES - Use most recent history including predictions
            # Extended lags: 1, 2, 3, 6, 12, 24, 48, 72, 168, 336
            lag_periods = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
            for lag in lag_periods:
                hour_features[f'lag_{lag}'] = history[-lag] if len(history) >= lag else last_vals['consumption'].mean()

            # Rolling windows
            for window in [24, 48, 168]:
                hour_features[f'roll_{window}'] = np.mean(history[-window:]) if len(history) >= window else last_vals['consumption'].mean()

            # Fill missing features
            for col in feature_cols:
                if col not in hour_features.columns:
                    hour_features[col] = 0

            # Predict for this hour using ENSEMBLE
            lgb_pred = lgb_model.predict(hour_features[feature_cols])[0]
            xgb_pred = xgb_model.predict(xgb.DMatrix(hour_features[feature_cols]))[0]

            # Ensemble: 60% LightGBM + 40% XGBoost
            pred = 0.6 * lgb_pred + 0.4 * xgb_pred
            group_predictions.append(pred)

            # UPDATE HISTORY with prediction for next iteration
            history.append(pred)
            if len(history) > 200:  # Keep history manageable
                history.pop(0)

        hourly_preds[str(group_id)] = group_predictions
    
    # Create submission
    hourly_sub = pd.DataFrame(hourly_preds)
    hourly_sub.insert(0, 'measured_at', future_hours.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
    
    print(f"✅ Hourly: {hourly_sub.shape}")
    
    # ==================
    # 8. MONTHLY FORECAST (12 months)
    # ==================
    print_section("MONTHLY PREDICTIONS")

    future_months = pd.date_range(start='2024-10-01', periods=12, freq='MS')

    # IMPROVED: Use seasonal patterns from historical data instead of random
    print("Calculating seasonal monthly patterns from historical data...")

    # Get monthly aggregates from historical data
    df_monthly = df.copy()
    df_monthly['year_month'] = df_monthly['measured_at'].dt.to_period('M')

    monthly_preds = {}
    for group_id in CUSTOMER_GROUPS:
        col = str(group_id)

        if col in df.columns:
            # Calculate monthly totals for each month in history
            monthly_totals = df_monthly.groupby('year_month')[col].sum()

            # Calculate seasonal factors by calendar month
            df_monthly['month'] = df_monthly['measured_at'].dt.month
            seasonal_factors = df_monthly.groupby('month')[col].sum() / df_monthly.groupby('month')[col].sum().mean()

            # Get base consumption from recent hourly predictions
            if col in hourly_sub.columns:
                base_hourly = hourly_sub[col].mean()
            else:
                base_hourly = monthly_totals.mean() / 730

            # Generate 12 monthly predictions with seasonal adjustment
            predictions = []
            for i, future_date in enumerate(future_months):
                month = future_date.month
                days_in_month = pd.Period(future_date, freq='M').days_in_month
                hours_in_month = days_in_month * 24

                # Apply seasonal factor if available
                seasonal_factor = seasonal_factors.get(month, 1.0)

                # Monthly prediction = hourly average × hours in month × seasonal factor
                monthly_pred = base_hourly * hours_in_month * seasonal_factor
                predictions.append(monthly_pred)

            monthly_preds[col] = predictions
        else:
            # Fallback for missing groups
            monthly_preds[col] = [1000 * 730] * 12

    monthly_sub = pd.DataFrame(monthly_preds)
    monthly_sub.insert(0, 'measured_at', future_months.strftime('%Y-%m-%dT%H:%M:%S.000Z'))

    print(f"✅ Monthly: {monthly_sub.shape}")
    
    # ==================
    # 9. SAVE (EUROPEAN FORMAT)
    # ==================
    print_section("SAVING SUBMISSIONS")
    
    hourly_sub.to_csv(CONFIG['output_hourly'], sep=';', decimal=',', index=False)
    monthly_sub.to_csv(CONFIG['output_monthly'], sep=';', decimal=',', index=False)
    
    print(f"✅ {CONFIG['output_hourly']}")
    print(f"✅ {CONFIG['output_monthly']}")
    
    print_section("COMPLETE!")
    print(f"Overall Model MAPE: {overall_mape:.2f}%")

    return models, overall_mape

if __name__ == "__main__":
    models, overall_mape = main()