import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score

def prepare_df():
    df = pd.read_excel('./Datasets/tobacco_data.xlsx')
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
                  'gender_idx', 'ethnicity_idx', 'qalys_pc']]
    return df_vape

def simple_duplicate(X, y, n_samples=200, random_state=None, noise_std=0.01):
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Determine how many times to duplicate the dataset
    n_repeats = n_samples // len(X)

    # Duplicate the data n_repeats times
    X_res = np.tile(X, (n_repeats, 1))
    y_res = np.tile(y, n_repeats)

    # Add Gaussian noise to the duplicated data
    noise = np.random.normal(0, 0.1, size=X_res.shape)
    X_res = X_res + noise

    return X_res, y_res

# Function to generate synthetic samples
def knn_samples(X, y, n_samples, random_state=42):
    np.random.seed(random_state)
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)

    synthetic_X = []
    synthetic_y = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(X))
        neighbors = nn.kneighbors([X[idx]], return_distance=False)[0]
        
        neighbor_idx = np.random.choice(neighbors)
        lam = np.random.uniform(0, 1)
        
        # Generate synthetic sample using interpolation
        new_sample_X = X[idx] + lam * (X[neighbor_idx] - X[idx])
        new_sample_y = y[idx] + lam * (y[neighbor_idx] - y[idx])
        
        synthetic_X.append(new_sample_X)
        synthetic_y.append(new_sample_y)
    
    return np.array(synthetic_X), np.array(synthetic_y)

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
    print("Best MAPE for Random Forest (No Bootstrap):", -grid_search_rf_no_bootstrap.best_score_)
    
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
    print("Best MAPE for Random Forest:", -grid_search_rf.best_score_)
    
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
    print("Best Parameters:", grid_search.best_params_)
    
    # Best MAPE score from cross-validation
    print("Best MAPE:", -grid_search.best_score_)
    
    # Train a final model using the best parameters
    best_xgb_model = grid_search.best_estimator_
    
    # Evaluate on the test set
    y_pred = best_xgb_model.predict(X_test)
    
    # Calculate the test MAPE
    test_mape = mape(y_test, y_pred)
    print("Test MAPE:", test_mape)

    return best_xgb_model, test_mape

if __name__ == "__main__":
    # Prepare DataFrame
    print("Preparing DataFrame")
    df_vape = prepare_df()
    columns = ['tax_increase', 'outlet_reduction', 'dec_smoking_prevalence', 
              'dec_tobacco_supply', 'dec_smoking_uptake', 'average_age', 
              'gender_idx', 'ethnicity_idx']
    X = df_vape[columns]
    y = df_vape[['qalys_pc']]
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    y_flat = y_train.values.flatten() # Ensure that y is a 1D array for compatibility
    
    # Generate synthetic samples
    print("Generating Synthetic Samples")
    X_sim, y_sim = simple_duplicate(X_train.values, y_flat, n_samples=200, random_state=42)
    X_sim = pd.DataFrame(X_sim, columns=columns)
    X_knn, y_knn = knn_samples(X_train.values, y_flat, n_samples=200, random_state=42)
    X_knn = pd.DataFrame(X_knn, columns=columns)
    X_reb, y_reb = knn_samples_rebalanced(X_train.values, y_flat, n_samples=200, random_state=42)
    X_reb = pd.DataFrame(X_reb, columns=columns)

    # Build Models
    print("Building No Bootstrap Model with Simple Duplication")
    no_bootstrap_sim_model, no_bootstrap_sim_test_mape = build_no_bootstrap(X_sim, X_test, y_sim, y_test, cv=5)
    print("")

    print("Building Bootstrap Model with Simple Duplication")
    bootstrap_sim_model, bootstrap_sim_test_mape = build_bootstrap(X_sim, X_test, y_sim, y_test, cv=5)
    print("")

    print("Building XGBoost Model with Simple Duplication")
    xgboost_sim_model, xgboost_sim_test_mape = build_xgboost(X_sim, X_test, y_sim, y_test, cv=5)
    print("")

    print("Building No Bootstrap Model with KNN Upsampling")
    no_bootstrap_knn_model, no_bootstrap_knn_test_mape = build_no_bootstrap(X_knn, X_test, y_knn, y_test, cv=5)
    print("")

    print("Building Bootstrap Model with KNN Upsampling")
    bootstrap_knn_model, bootstrap_knn_test_mape = build_bootstrap(X_knn, X_test, y_knn, y_test, cv=5)
    print("")

    print("Building XGBoost Model with KNN Upsampling")
    xgboost_knn_model, xgboost_knn_test_mape = build_xgboost(X_knn, X_test, y_knn, y_test, cv=5)
    print("")
    
    print("Building No Bootstrap Model with Rebalanced KNN Upsampling")
    no_bootstrap_reb_model, no_bootstrap_reb_test_mape = build_no_bootstrap(X_reb, X_test, y_reb, y_test, cv=5)
    print("")

    print("Building Bootstrap Model with Rebalanced KNN Upsampling")
    bootstrap_reb_model, bootstrap_reb_test_mape = build_bootstrap(X_reb, X_test, y_reb, y_test, cv=5)
    print("")

    print("Building XGBoost Model with Rebalanced KNN Upsampling")
    xgboost_reb_model, xgboost_reb_test_mape = build_xgboost(X_reb, X_test, y_reb, y_test, cv=5)
    print("")
    