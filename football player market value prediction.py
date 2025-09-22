# Football Player Market Value Prediction Model

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load dataset
df = pd.read_csv("train.csv")

# Focus on the player's main role 
df['primary_position'] = df['player_positions'].fillna('').apply(lambda x: x.split(',')[0].strip())

# Convert categorical data into standardized body type classification
df['body_type_simplified'] = df['body_type'].fillna('').apply(
    lambda x: 'Lean' if 'Lean' in x else 'Stocky' if 'Stocky' in x else 'Normal' if 'Normal' in x else 'Other'
)

# One-hot encoded features for body type and positions
body_dummies = pd.get_dummies(df['body_type_simplified'], prefix='body_type')
position_dummies = pd.get_dummies(df['primary_position'], prefix='pos')
df = pd.concat([df, body_dummies, position_dummies], axis=1)

# Compute body type correlation score with market value
bt_cols = body_dummies.columns
bt_weights = {col: df[[col, 'value_eur']].corr().iloc[0, 1] for col in bt_cols}
df['body_type_score'] = df[bt_cols].mul(pd.Series(bt_weights)).sum(axis=1)

# Position group definitions
group_definitions = {
    "fullbacks": ['LB', 'RB', 'RWB', 'LWB'],
    "center_backs": ['CB'],
    "midfielders": ['CM', 'CDM'],
    "wingers": ['LM', 'RM', 'RW', 'LW'],
    "forwards": ['ST', 'CAM', 'CF'],
    "goalkeepers": ['GK']
}

# Categorize features 
goalkeeping_features = [col for col in df.columns if col.startswith('goalkeeping_')]
outfield_features = [col for col in df.columns if col.startswith((
    'att_', 'def_', 'skill_', 'mentality_', 'power_', 'movement_', 'passing',
    'shooting', 'dribbling', 'physic', 'pace', 'overall', 'potential', 'age',
    'wage_eur', 'height_cm', 'weight_kg', 'club_team_id', 'league_level',
    'club_jersey_number', 'club_contract_valid_until'))]

# Feature selection
# Returns a list of feature columns that meet the correlation threshold
def select_features(df, target, feature_cols, threshold=0.05):
    corrs = df[feature_cols + [target]].corr()[target].abs()
    selected = corrs[corrs > threshold].index.tolist()
    if target in selected:
        selected.remove(target)
    return selected

# Position-Specific Model Training

#Store model results for each position group
group_results = {}

for group_name, positions in group_definitions.items():
    #print(f"\n Training XGBoost model for: {group_name} ({positions})")
    
    # Filter for players in current position group
    group_df = df[df['primary_position'].isin(positions)].copy()
    group_df = group_df[group_df['value_eur'] > 0]

    # Apply winsorization to reduce impact of extreme outliers
    # Cap at 99.5 percentile
    cap = group_df['value_eur'].quantile(0.995)
    group_df['value_eur'] = np.minimum(group_df['value_eur'], cap)
    print(f"Capped value_eur at €{cap:,.0f} to reduce skew.")

    # Normalize distribution log-transforming the interested variable
    group_df['log_value'] = np.log1p(group_df['value_eur'])
    
    # Interaction features
    group_df['age_potential'] = group_df['age'] * group_df['potential']
    group_df['value_league'] = group_df['value_eur'] * group_df['league_level']
    group_df['bodytype_position'] = group_df['body_type_score'] * position_dummies.loc[group_df.index].sum(axis=1)
    # Nuove feature (train)
    # group_df['is_star'] = (group_df['overall'] >= 88).astype(int)
    # group_df['overall_potential'] = group_df['overall'] * group_df['potential']
    # group_df['age_inverse'] = 1 / group_df['age'].replace(0, np.nan)
    
     # Select base features group
    base_features = goalkeeping_features if group_name == "goalkeepers" else outfield_features
    
    # Select interaction features
    interaction_features = ['body_type_score', 'age_potential', 'value_league', 'bodytype_position']
    # top players
    # interaction_features = ['body_type_score','age_potential','value_league','bodytype_position','is_star','overall_potential','age_inverse']
    
    # Add them together
    all_features = interaction_features + base_features

    # Exlude irrelevant features for each position
    exclude_by_group = {
        "fullbacks": ['shooting', 'attacking_volleys', 'attacking_heading_accuracy', 'club_jersey_number', 'skill_fk_accuracy', 'movement_agility', 'mentality_vision', 'power_jumping'] +  goalkeeping_features,
        "center_backs": ['dribbling', 'skill_curve', 'attacking_crossing', 'mentality_vision'] +  goalkeeping_features,
        "midfielders": ['attacking_heading_accuracy'] + goalkeeping_features,
        "wingers": ['attacking_heading_accuracy', 'defending_standing_tackle'] +  goalkeeping_features,
        "forwards": ['mentality_interceptions', 'defending_marking_awareness', 'defending_sliding_tackle'] +  goalkeeping_features,
        "goalkeepers": outfield_features + ['goalkeeping_speed']
    }

    # Remove position-irrelevant features 
    to_exclude = exclude_by_group.get(group_name, [])
    all_features = [f for f in all_features if f not in to_exclude]

    # Handle missing values
    group_df = group_df.replace([np.inf, -np.inf], np.nan).dropna(subset=all_features + ['log_value'])
    selected = select_features(group_df, 'log_value', all_features, threshold=0.05)

    # Define feature matrix and target value
    X = group_df[selected]
    y = group_df['log_value']

    # Initialize 5-fold cross-validation 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to collect performance metrics 
    fold_train_errors = []
    fold_validation_errors = []
    
    # Variables to track the best model
    best_model = None
    best_val_rmse = float('inf')

    # Train model for each cross-validation fold
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize XGBoost regressor 
        # These parameters are chosen to prevent overfitting and handle complexity
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Train the model 
        model.fit(X_train, y_train)

        # Generate predictions 
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate metrics - RMSE
        train_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(y_train_pred)))
        val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_val_pred)))

         # Add this fold's metrics to our running lists of all fold results
        fold_train_errors.append(train_rmse)
        fold_validation_errors.append(val_rmse)
        
        # Keep track of the best performing model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model

    avg_train_rmse = np.mean(fold_train_errors)
    avg_val_rmse = np.mean(fold_validation_errors)
    gap = avg_val_rmse - avg_train_rmse

    # print(f"{group_name.capitalize()} - Train RMSE: €{avg_train_rmse:,.2f}, Validation RMSE: €{avg_val_rmse:,.2f}, Gap: €{gap:,.2f}")

    group_results[group_name] = {
        "model": best_model,
        "train_rmse": avg_train_rmse,
        "val_rmse": avg_val_rmse,
        "gap": gap,
        "features": selected
    }

 
    # === TEST SET ===
test_df = pd.read_csv("test.csv")
submission_df = pd.read_csv("submission.csv")

# Preprocessing base
test_df['primary_position'] = test_df['player_positions'].fillna('').apply(lambda x: x.split(',')[0].strip())
test_df['body_type_simplified'] = test_df['body_type'].fillna('').apply(
    lambda x: 'Lean' if 'Lean' in x else 'Stocky' if 'Stocky' in x else 'Normal' if 'Normal' in x else 'Other'
)
body_dummies = pd.get_dummies(test_df['body_type_simplified'], prefix='body_type')
position_dummies = pd.get_dummies(test_df['primary_position'], prefix='pos')
test_df = pd.concat([test_df, body_dummies, position_dummies], axis=1)

bt_cols = body_dummies.columns
# Usa gli stessi pesi calcolati nel training set
bt_weights = {col: df[[col, 'value_eur']].corr().iloc[0, 1] for col in bt_cols if col in df.columns}

test_df['body_type_score'] = test_df[bt_cols].mul(pd.Series(bt_weights)).sum(axis=1)

# Predict
predictions = []

for group_name, positions in group_definitions.items():
    if group_name not in group_results:
        continue
    model = group_results[group_name]["model"]
    selected = group_results[group_name]["features"]

    group_test = test_df[test_df['primary_position'].isin(positions)].copy()
    if group_test.empty:
        continue

    group_test['age_potential'] = group_test['age'] * group_test['potential']
    group_test['value_league'] = group_test['league_level']
    group_test['bodytype_position'] = group_test['body_type_score'] * position_dummies.loc[group_test.index].sum(axis=1)
    # Nuove feature (test)
    # group_test['is_star'] = (group_test['overall'] >= 88).astype(int)
    # group_test['overall_potential'] = group_test['overall'] * group_test['potential']
    # group_test['age_inverse'] = 1 / group_test['age'].replace(0, np.nan)

    group_test = group_test.replace([np.inf, -np.inf], np.nan).dropna(subset=selected)
    X_test = group_test[selected]

    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred)

    result_df = pd.DataFrame({
        'index': group_test['Unnamed: 0'],  # uso l'indice originale per ordinare
        'value_eur': y_pred.round().astype(float)
    })


    predictions.append(result_df)

# Final merge and save
final_submission = pd.concat(predictions, ignore_index=True)

# Ordina per indice originale per rispettare l’ordine da 0 a 3847
final_submission = final_submission.sort_values(by="index").reset_index(drop=True)

# Salva solo la colonna value_eur, con index salvato (Unnamed: 0)
final_submission[['value_eur']].to_csv(
    "submission_final.csv",
    index=True
)

print("✅ File submission_final.csv salvato correttamente con index + value_eur.")




# Plot importances after training all models
for group_name, result in group_results.items():
    model = result["model"]
    feature_names = result["features"]

    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
    importance_df.index.name = 'feature'
    importance_df.reset_index(inplace=True)

    low_importance_df = importance_df[importance_df['gain'] <= 0.02].sort_values(by='gain')

    if low_importance_df.empty:
        print(f"\n No low-importance features (≤ 0.02 gain) for {group_name}.")
        continue

    print(f"\n Low-importance features for {group_name.capitalize()}")
    plt.figure(figsize=(8, max(4, len(low_importance_df) * 0.5)))
    plt.barh(low_importance_df['feature'], low_importance_df['gain'], color='skyblue')
    plt.xlabel("Gain (Importance)")
    plt.title(f"{group_name.capitalize()} - Low Importance Features (≤ 0.02)")
    plt.tight_layout()
    plt.show()