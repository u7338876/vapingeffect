import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def prepare_df():
    df = pd.read_excel('./Datasets/tobacco_data_v2.xlsx')
    df.columns = df.iloc[0]
    df = df[1:]
    
    # Map age group to integer
    avg_age_mapping = {
        '0-14': 7,
        '15-24': 20,
        '25-44': 33,
        '45-64': 55,
        '65+': 75
    }
    
    # Map gender to integer
    gender_mapping = {
        'Male': 0,
        'Female': 1
    }

    # Map ethnicity to integer
    ethnicity_mapping = {
        'Māori': 0,
        'non-Māori': 1
    }
    
    # Apply the mapping to the 'Age_Group' column
    df['average_age'] = df['age'].map(avg_age_mapping)
    df['gender_idx'] = df['gender'].map(gender_mapping)
    df['ethnicity_idx'] = df['ethnicity'].map(ethnicity_mapping)
    
    # Impute missing values in 'average_age' with the mean
    df['average_age'] = df['average_age'].fillna(df['average_age'].mean())
    
    # Impute missing values in 'gender_idx' and 'ethnicity_idx' with the mode
    df['gender_idx'] = df['gender_idx'].fillna(df['gender_idx'].mode()[0])
    df['ethnicity_idx'] = df['ethnicity_idx'].fillna(df['ethnicity_idx'].mode()[0])
    
    # Convert the specified columns to floats
    df[['tax_increase', 'outlet_reduction', 'dec_smoking_prevalence', 
        'dec_tobacco_supply', 'dec_smoking_uptake', 'qalys_pc']] = df[['tax_increase', 'outlet_reduction', 
        'dec_smoking_prevalence', 'dec_tobacco_supply', 'dec_smoking_uptake', 'qalys_pc']].apply(pd.to_numeric, errors='coerce').astype('float')
    
    # Columns to be used for model building
    df_vape = df[['tax_increase', 'outlet_reduction', 'dec_smoking_prevalence', 
                  'dec_tobacco_supply', 'dec_smoking_uptake', 'average_age', 
                  'gender_idx', 'ethnicity_idx', 'qalys_pc', 'hs_costs_pc']]
    return df_vape

def prepare_df_vape():
    df_vape = pd.read_excel('./Datasets/vaping_data.xlsx')

    # Map age group to integer
    avg_age_mapping = {
        '0-14': 7,
        '15-24': 20,
        '25-44': 33,
        '45-64': 55,
        '65+': 75
    }
    
    # Map gender to integer
    gender_mapping = {
        'Male': 0,
        'Female': 1
    }

    # Map ethnicity to integer
    ethnicity_mapping = {
        'Māori': 0,
        'non-Māori': 1
    }
    
    # Apply the mapping to the 'Age_Group' column
    df_vape['average_age'] = df_vape['age'].map(avg_age_mapping)
    df_vape['gender_idx'] = df_vape['gender'].map(gender_mapping)
    df_vape['ethnicity_idx'] = df_vape['ethnicity'].map(ethnicity_mapping)

    return df_vape

def knn_samples_rebalanced(X, y, n_samples, threshold=0.2, random_state=42):
    np.random.seed(random_state)
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)

    # Calculate distances to the nearest neighbors
    distances, _ = nn.kneighbors(X)

    # Calculate the number of neighbors within the threshold for each sample
    neighbors_within_threshold = np.sum(distances[:, 1:] < threshold, axis=1)

    # Calculate inverse probabilities
    selection_probabilities = 1 / (neighbors_within_threshold + 1e-6)  # Add small constant to avoid division by zero
    selection_probabilities /= selection_probabilities.sum()  # Normalize to sum to 1

    synthetic_X = []
    synthetic_y = []

    for _ in range(n_samples):
        # Select a sample based on the weighted probability distribution
        idx = np.random.choice(len(X), p=selection_probabilities)

        # Select a random neighbor of the chosen sample
        neighbors = nn.kneighbors([X[idx]], return_distance=False)[0]
        neighbor_idx = np.random.choice(neighbors)

        # Linear interpolation for synthetic sample generation
        lam = np.random.uniform(0, 1)
        new_sample_X = X[idx] + lam * (X[neighbor_idx] - X[idx])
        new_sample_y = y[idx] + lam * (y[neighbor_idx] - y[idx])

        synthetic_X.append(new_sample_X)
        synthetic_y.append(new_sample_y)

    # Stack the original and synthetic data
    X_full = np.vstack([X_train.values, synthetic_X])
    y_full = np.hstack([y_flat, synthetic_y])
    
    return np.array(X_full), np.array(y_full)

def build_no_bootstrap(X_train, X_test, y_train, y_test, cv=5):    
    # Define the RandomForestRegressor model with bootstrap disabled
    rf_model_no_bootstrap = RandomForestRegressor(random_state=42, bootstrap=False)
    
    # Define the parameter grid to search over
    param_grid_no_bootstrap = {
        'n_estimators': [100, 200, 300],    # Number of trees in the forest
        'max_depth': [3, 5, 10],            # Maximum depth of the tree
        'min_samples_leaf': [1, 2, 4],      # Minimum number of samples required to be at a leaf node
    }
    
    # Define the MAPE scorer (using Mean Absolute Percentage Error)
    mape_scorer = make_scorer(mape, greater_is_better=False)
    
    # Setup GridSearchCV to perform cross-validation
    grid_search_rf_no_bootstrap = GridSearchCV(estimator=rf_model_no_bootstrap, param_grid=param_grid_no_bootstrap, 
                                               scoring=mape_scorer, cv=5, verbose=1, n_jobs=-1)
    
    # Fit the grid search to the duplicated training data
    grid_search_rf_no_bootstrap.fit(X_train, y_train)
    
    # Best hyperparameters from grid search
    print("Best Parameters for Random Forest (No Bootstrap):", grid_search_rf_no_bootstrap.best_params_)
    
    # Best MAPE score from cross-validation
    print("Best CV MAPE for Random Forest (No Bootstrap):", -grid_search_rf_no_bootstrap.best_score_)
    
    # Train a final model using the best parameters
    best_no_bootstrap_model = grid_search_rf_no_bootstrap.best_estimator_
    
    # Evaluate on the test set
    y_pred_rf_no_bootstrap = best_no_bootstrap_model.predict(X_test)
    
    # Calculate the test MAPE
    test_mape = mape(y_test, y_pred_rf_no_bootstrap)
    print("Test MAPE for Random Forest (No Bootstrap):", test_mape)

    return best_no_bootstrap_model, test_mape

def build_bootstrap(X_train, X_test, y_train, y_test, cv=5):    
    # Define the RandomForestRegressor model
    rf_model = RandomForestRegressor(random_state=42, bootstrap=True)
    
    # Define the parameter grid to search over
    param_grid = {
        'n_estimators': [100, 200, 300],    # Number of trees in the forest
        'max_depth': [5, 10, 20],            # Maximum depth of the tree
        'min_samples_leaf': [1, 5, 10],      # Minimum number of samples required to be at a leaf node
        'max_samples': [0.5, 0.7, 1.0],     # Maximum number of samples to draw from the data with replacement
    }
    
    # Define the MAPE scorer (using Mean Absolute Percentage Error)
    mape_scorer = make_scorer(mape, greater_is_better=False)
    
    # Setup GridSearchCV to perform cross-validation
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                                  scoring=mape_scorer, cv=5, verbose=1, n_jobs=-1)
    
    # Fit the grid search to the duplicated training data
    grid_search_rf.fit(X_train, y_train)
    
    # Best hyperparameters from grid search
    print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
    
    # Best MAPE score from cross-validation
    print("Best CV MAPE for Random Forest:", -grid_search_rf.best_score_)
    
    # Train a final model using the best parameters
    best_bootstrap_model = grid_search_rf.best_estimator_
    
    # Evaluate on the test set
    y_pred_rf = best_bootstrap_model.predict(X_test)
    
    # Calculate the test MAPE
    test_mape = mape(y_test, y_pred_rf)
    print("Test MAPE for Random Forest:", test_mape)

    return best_bootstrap_model, test_mape

def build_xgboost(X_train, X_test, y_train, y_test, cv=5):
    # Define the XGBoost model
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Define the parameter grid to search over
    param_grid = {
        'n_estimators': [100, 200, 300],   # Number of trees
        'max_depth': [5, 10, 20],            # Depth of the trees
        'min_child_weight': [1, 5, 10],     # Minimum sum of instance weight (hessian)
        'reg_lambda': [0.01, 0.1, 1, 10],  # L2 regularization term (lambda)
        'reg_alpha': [0.01, 0.1, 1, 10],      # L1 regularization term (alpha)
    }
    
    # Define the MAPE scorer (as we are optimizing based on Mean Absolute Percentage Error)
    mape_scorer = make_scorer(mape, greater_is_better=False)
    
    # Setup GridSearchCV to perform cross-validation
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring=mape_scorer, cv=5, verbose=1, n_jobs=-1)
    
    # Fit the grid search to the duplicated training data
    grid_search.fit(X_train, y_train)
    
    # Best hyperparameters from grid search
    print("Best Parameters for XGBoost:", grid_search.best_params_)
    
    # Best MAPE score from cross-validation
    print("Best CV MAPE for XGBoost:", -grid_search.best_score_)
    
    # Train a final model using the best parameters
    best_xgb_model = grid_search.best_estimator_
    
    # Evaluate on the test set
    y_pred = best_xgb_model.predict(X_test)
    
    # Calculate the test MAPE
    test_mape = mape(y_test, y_pred)
    print("Test MAPE for XGBoost:", test_mape)

    return best_xgb_model, test_mape

if __name__ == "__main__":
    # Prepare datasets
    df = prepare_df()
    columns = ['tax_increase', 'outlet_reduction', 'dec_smoking_prevalence', 
                  'dec_tobacco_supply', 'dec_smoking_uptake', 'average_age', 
                  'gender_idx', 'ethnicity_idx']
    
    X_train = df[columns]
    y_train = df[['qalys_pc']]

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    y_flat = y_train.values.flatten() # Ensure that y is a 1D array for compatibility
    
    X_reb, y_reb = knn_samples_rebalanced(X_train.values, y_flat, n_samples=200, random_state=42)
    X_reb = pd.DataFrame(X_reb, columns=columns)

    df_vape = prepare_df_vape()
    pred = df_vape.drop(columns=['age', 'gender', 'ethnicity'])

    # QALY Prepare Models
    # 1. Random Forest (No Bootstrap)
    qaly_rf_no_bootstrap = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=42
    )
    
    # 2. Random Forest (With Bootstrap)
    qaly_rf_bootstrap = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=1,
        bootstrap=True,    # default is True, but specifying for clarity
        max_samples=1.0,   # 1.0 = 100% sampling
        random_state=42
    )
    
    # 3. XGBoost (No Bootstrap)
    qaly_xgb_no_bootstrap = XGBRegressor(
        n_estimators=100,
        max_depth=10,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=0.01,
        objective='reg:squarederror',  # to avoid warnings
        random_state=42
    )
    
    # 4. XGBoost (With Bootstrap)
    qaly_xgb_bootstrap = XGBRegressor(
        n_estimators=300,
        max_depth=20,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        sampling_method='uniform',
        subsample=0.999,
        objective='reg:squarederror',
        random_state=42
    )
    
    qaly_stacking = StackingRegressor(
        cv=5,
        estimators=[
            ('ridge', Ridge()),
            ('lasso', Lasso(alpha=0.1)),
            ('svr', SVR(kernel='linear')),
            ('tree', DecisionTreeRegressor(max_depth=5)),
            ('rf', RandomForestRegressor(random_state=42))
        ],
        final_estimator=Lasso(alpha=0.1)
    )

    # Train QALY Model
    qaly_rf_no_bootstrap.fit(X_reb, y_reb)
    qaly_rf_no_bootstrap_pred = qaly_rf_no_bootstrap.predict(pred)
    
    qaly_rf_bootstrap.fit(X_reb, y_reb)
    qaly_rf_bootstrap_pred = qaly_rf_bootstrap.predict(pred)
    
    qaly_xgb_no_bootstrap.fit(X_reb, y_reb)
    qaly_xgb_no_bootstrap_pred = qaly_xgb_no_bootstrap.predict(pred)
    
    qaly_xgb_bootstrap.fit(X_reb, y_reb)
    qaly_xgb_bootstrap_pred = qaly_xgb_bootstrap.predict(pred)
    
    qaly_stacking.fit(X_reb, y_reb)
    qaly_stacking_pred = qaly_stacking.predict(pred)
    
    qaly_pred = (qaly_rf_no_bootstrap_pred + qaly_rf_bootstrap_pred + qaly_xgb_no_bootstrap_pred + qaly_xgb_bootstrap_pred + qaly_stacking_pred) / 5
    pred['qalys_pc'] = qaly_pred

    # Train HSC Model
    columns.append('qalys_pc')
    X_train = df[columns]
    y_train = df[['hs_costs_pc']]

    y_flat = y_train.values.flatten() # Ensure that y is a 1D array for compatibility
    X_reb, y_reb = knn_samples_rebalanced(X_train.values, y_flat, n_samples=200, random_state=42)
    X_reb = pd.DataFrame(X_reb, columns=columns)

    # Prepare HSC Models
    hsc_rf_no_bootstrap = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=1,
        n_estimators=300,
        bootstrap=False,        # No bootstrapping
        random_state=42,
        n_jobs=-1
    )
        
        
    hsc_rf_bootstrap = RandomForestRegressor(
        max_depth=20,
        max_samples=1.0,         # Using all samples
        min_samples_leaf=1,
        n_estimators=300,
        bootstrap=True,          # Bootstrapping enabled
        random_state=42,
        n_jobs=-1
    )
        
        
    hsc_xgb_no_bootstrap = XGBRegressor(
        max_depth=20,
        min_child_weight=1,
        n_estimators=100,
        reg_alpha=0.1,
        reg_lambda=1,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    hsc_xgb_bootstrap = XGBRegressor(
        max_depth=20,
        min_child_weight=1,
        n_estimators=300,
        reg_alpha=10,
        reg_lambda=0.1,
        subsample=0.75,
        sampling_method='uniform',    # uniform sampling
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Base learners
    base_learners = [
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.1)),
        ('svr', SVR(kernel='linear', C=1.0)),
        ('tree', DecisionTreeRegressor(max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    
    # Meta learner
    meta_learner = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # Stacking ensemble
    hsc_stacking = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    hsc_rf_no_bootstrap.fit(X_reb, y_reb)
    hsc_rf_no_bootstrap_pred = hsc_rf_no_bootstrap.predict(pred)
    
    hsc_rf_bootstrap.fit(X_reb, y_reb)
    hsc_rf_bootstrap_pred = hsc_rf_bootstrap.predict(pred)
    
    hsc_xgb_no_bootstrap.fit(X_reb, y_reb)
    hsc_xgb_no_bootstrap_pred = hsc_xgb_no_bootstrap.predict(pred)
    
    hsc_xgb_bootstrap.fit(X_reb, y_reb)
    hsc_xgb_bootstrap_pred = hsc_xgb_bootstrap.predict(pred)
    
    hsc_stacking.fit(X_reb, y_reb)
    hsc_stacking_pred = hsc_stacking.predict(pred)
    
    hsc_pred = (hsc_rf_no_bootstrap_pred + hsc_rf_bootstrap_pred + hsc_xgb_no_bootstrap_pred + hsc_xgb_bootstrap_pred + hsc_stacking_pred) / 5

    pred['hs_costs_pc'] = hsc_pred
    result = df_vape[['age', 'gender', 'ethnicity']].copy()
    result['QALYs'] = pred['qalys_pc']
    result['HSCs'] = pred['hs_costs_pc']
    print(result)

    # File path
    file_path = "./Datasets/results_full.xlsx"

    # Save to Excel (overwrite mode)
    result.to_excel(file_path, index=False, engine='openpyxl')

    print("Results Saved")



    
