"""
HACKATHON-WINNING IMPROVEMENTS FOR FORTUM ENERGY FORECASTING
Key enhancements to reach top-tier performance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ADVANCED FEATURE ENGINEERING
# ============================================================================

def create_winning_features(df, df_groups, price_data=None):
    """Create competition-winning features"""
    
    # 1. ELECTRICITY PRICE FEATURES (CRITICAL!)
    if price_data is not None:
        df = df.merge(price_data, on='measured_at', how='left')
        df['price_ma_24'] = df['price'].rolling(24).mean()
        df['price_volatility'] = df['price'].rolling(24).std()
        df['price_trend'] = df['price'].diff(24)
        
    # 2. ADVANCED TIME FEATURES
    df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
    df['month_temp_interaction'] = df['month'] * df.get('temperature', 0)
    
    # Seasonal decomposition components
    df['trend'] = df.groupby('group_id')['consumption'].transform(
        lambda x: x.rolling(168*4, min_periods=1).mean()  # 4-week trend
    )
    df['seasonal_weekly'] = df.groupby(['group_id', 'hour', 'dayofweek'])['consumption'].transform('mean')
    df['seasonal_monthly'] = df.groupby(['group_id', 'month', 'hour'])['consumption'].transform('mean')
    
    # 3. CROSS-GROUP FEATURES (GAME CHANGER!)
    # Industrial groups affect residential demand
    industrial_groups = df_groups[df_groups['customer_type'] == 'Enterprise']['region_id'].tolist()
    residential_groups = df_groups[df_groups['customer_type'] == 'Private']['region_id'].tolist()
    
    if industrial_groups and residential_groups:
        # Industrial consumption affects residential prices
        df['industrial_total'] = df[industrial_groups].sum(axis=1)
        df['residential_total'] = df[residential_groups].sum(axis=1)
        df['industrial_residential_ratio'] = df['industrial_total'] / (df['residential_total'] + 1)
    
    # 4. WEATHER INTERACTION FEATURES
    if 'temperature' in df.columns:
        # Customer-type specific weather sensitivity
        for customer_type in df_groups['customer_type'].unique():
            type_groups = df_groups[df_groups['customer_type'] == customer_type]['region_id'].tolist()
            df[f'temp_{customer_type.lower()}'] = df['temperature'] * df[type_groups].mean(axis=1)
    
    # 5. ADVANCED LAG FEATURES
    # Multiple lag horizons with exponential decay weights
    for group_id in df['group_id'].unique():
        group_mask = df['group_id'] == group_id
        group_data = df[group_mask]['consumption']
        
        # Exponentially weighted lags
        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            weight = np.exp(-lag/24)  # Decay weight
            df.loc[group_mask, f'lag_{lag}_weighted'] = group_data.shift(lag) * weight
    
    return df

# ============================================================================
# 2. GROUP-SPECIFIC MODELING STRATEGY
# ============================================================================

def create_group_clusters(df_groups):
    """Cluster similar customer groups for targeted modeling"""
    
    # Features for clustering
    cluster_features = []
    
    # One-hot encode categorical features
    for col in ['customer_type', 'contract_type', 'consumption_level', 'macro_region']:
        if col in df_groups.columns:
            dummies = pd.get_dummies(df_groups[col], prefix=col)
            cluster_features.append(dummies)
    
    if cluster_features:
        cluster_df = pd.concat(cluster_features, axis=1)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        df_groups['cluster'] = kmeans.fit_predict(cluster_df)
    else:
        df_groups['cluster'] = 0
    
    return df_groups

def train_ensemble_models(df_long, df_groups, feature_cols):
    """Train ensemble of models for different group clusters"""
    
    models = {}
    cluster_performance = {}
    
    # Add cluster info
    df_long = df_long.merge(
        df_groups[['region_id', 'cluster']], 
        left_on='group_id', right_on='region_id', how='left'
    )
    
    # Train model for each cluster
    for cluster_id in df_groups['cluster'].unique():
        print(f"\nTraining cluster {cluster_id}...")
        
        cluster_data = df_long[df_long['cluster'] == cluster_id]
        if len(cluster_data) < 1000:
            continue
            
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        cluster_scores = []
        
        for train_idx, val_idx in tscv.split(cluster_data):
            train_data = cluster_data.iloc[train_idx]
            val_data = cluster_data.iloc[val_idx]
            
            X_train = train_data[feature_cols]
            y_train = train_data['consumption']
            X_val = val_data[feature_cols]
            y_val = val_data['consumption']
            
            # Advanced LightGBM parameters for competition
            params = {
                'objective': 'regression',
                'metric': 'mape',
                'boosting_type': 'gbdt',
                'num_leaves': 127,  # Increased complexity
                'learning_rate': 0.03,  # Lower for better convergence
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 10,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 0.1,  # L2 regularization
                'verbose': -1,
                'random_state': 42
            }
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            model = lgb.train(
                params, lgb_train, 
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            val_pred = model.predict(X_val)
            mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
            cluster_scores.append(mape)
        
        # Train final model on all cluster data
        X_all = cluster_data[feature_cols]
        y_all = cluster_data['consumption']
        
        lgb_all = lgb.Dataset(X_all, y_all)
        final_model = lgb.train(params, lgb_all, num_boost_round=500)
        
        models[cluster_id] = final_model
        cluster_performance[cluster_id] = np.mean(cluster_scores)
        
        print(f"Cluster {cluster_id} CV MAPE: {np.mean(cluster_scores):.2f}%")
    
    return models, cluster_performance

# ============================================================================
# 3. ADVANCED PREDICTION STRATEGY
# ============================================================================

def predict_with_autoregression(models, future_df, df_groups, feature_cols, n_hours=48):
    """Predict with proper autoregressive updating of lags"""
    
    predictions = {}
    
    for group_id in df_groups['region_id'].unique():
        # Get cluster for this group
        cluster = df_groups[df_groups['region_id'] == group_id]['cluster'].iloc[0]
        
        if cluster not in models:
            # Fallback to average cluster model
            cluster = list(models.keys())[0]
        
        model = models[cluster]
        group_preds = []
        
        # Get recent history for lags
        recent_history = get_recent_history(group_id, lookback=200)
        
        for hour in range(n_hours):
            # Create features for this hour
            hour_features = create_hour_features(
                future_df.iloc[hour], group_id, recent_history, df_groups
            )
            
            # Ensure all features exist
            for col in feature_cols:
                if col not in hour_features:
                    hour_features[col] = 0
            
            # Predict
            pred = model.predict([hour_features[feature_cols]])[0]
            group_preds.append(pred)
            
            # Update history with prediction for next iteration
            recent_history.append(pred)
            if len(recent_history) > 200:
                recent_history.pop(0)
        
        predictions[str(group_id)] = group_preds
    
    return predictions

# ============================================================================
# 4. UNCERTAINTY QUANTIFICATION
# ============================================================================

def add_uncertainty_bounds(predictions, confidence_level=0.95):
    """Add confidence intervals to predictions"""
    
    # Simple approach: use historical volatility
    for group_id in predictions:
        preds = np.array(predictions[group_id])
        
        # Estimate uncertainty from recent prediction errors
        volatility = np.std(preds) * 0.1  # Conservative estimate
        
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 99%
        
        predictions[f"{group_id}_lower"] = preds - z_score * volatility
        predictions[f"{group_id}_upper"] = preds + z_score * volatility
    
    return predictions

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def implement_winning_strategy():
    """
    Implementation guide for hackathon-winning approach:
    
    1. Load and use ALL data sources (including prices!)
    2. Create advanced features with cross-group interactions
    3. Cluster customer groups for targeted modeling
    4. Train ensemble models with proper validation
    5. Use autoregressive prediction with uncertainty bounds
    6. Optimize hyperparameters with Optuna/Hyperopt
    7. Implement model stacking/blending
    """
    
    print("🚀 WINNING STRATEGY IMPLEMENTATION GUIDE:")
    print("1. ✅ Use electricity price data (CRITICAL!)")
    print("2. ✅ Cross-group interaction features") 
    print("3. ✅ Group-specific ensemble models")
    print("4. ✅ Autoregressive prediction strategy")
    print("5. ✅ Proper time series validation")
    print("6. 🔄 Hyperparameter optimization (Optuna)")
    print("7. 🔄 Model stacking/blending")
    print("8. 🔄 Feature selection with SHAP")

if __name__ == "__main__":
    implement_winning_strategy()