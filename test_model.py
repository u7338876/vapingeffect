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
def generate_synthetic_samples(X, y, n_samples, random_state=42):
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

def generate_synthetic_samples_imbalanced(X, y, n_samples, threshold=0.2, random_state=42):
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

    return np.array(synthetic_X), np.array(synthetic_y)

def build_model(X_train, X_test, y_train, y_test, cv=5):    
        # Define the RandomForestRegressor model with bootstrap disabled
    rf_model_no_bootstrap = RandomForestRegressor(random_state=42, bootstrap=False)

    # Define the parameter grid to search over
    param_grid_no_bootstrap = {
        'n_estimators': [100, 200, 300],    # Number of trees in the forest
        'max_depth': [3, 5, 10],            # Maximum depth of the tree
        'min_samples_leaf': [1, 2, 4],      # Minimum samples required at a leaf node
    }

    # Define the MAPE scorer (using Mean Absolute Percentage Error)
    mape_scorer = make_scorer(mape, greater_is_better=False)

    # Convert y_train to a 1D array using NumPy's ravel()
    y_train_1d = np.ravel(y_train)

    # Setup GridSearchCV to perform cross-validation
    grid_search_rf_no_bootstrap = GridSearchCV(
        estimator=rf_model_no_bootstrap, 
        param_grid=param_grid_no_bootstrap, 
        scoring=mape_scorer, 
        cv=cv, 
        verbose=1, 
        n_jobs=-1
    )

    # Fit the grid search to the training data
    grid_search_rf_no_bootstrap.fit(X_train, y_train_1d)

    # Get the best hyperparameters and best score from grid search
    best_rf_model_no_bootstrap = grid_search_rf_no_bootstrap.best_estimator_
    best_params_rf_no_bootstrap = grid_search_rf_no_bootstrap.best_params_
    best_cv_mape_no_bootstrap = -grid_search_rf_no_bootstrap.best_score_

    print("Best Parameters for Random Forest (No Bootstrap):", best_params_rf_no_bootstrap)
    print("Best CV MAPE for Random Forest (No Bootstrap):", best_cv_mape_no_bootstrap)

    # Perform cross-validation with the best model to get average MAPE
    cv_mape_scores_no_bootstrap = cross_val_score(
        best_rf_model_no_bootstrap, 
        X_train, 
        y_train_1d, 
        scoring=mape_scorer, 
        cv=cv, 
        n_jobs=-1
    )
    avg_cv_mape_no_bootstrap = -cv_mape_scores_no_bootstrap.mean()

    print(f"Average CV MAPE for Random Forest (No Bootstrap): {avg_cv_mape_no_bootstrap}")

    # Train the final model using the best parameters
    best_rf_model_no_bootstrap.fit(X_train, y_train_1d)

    # Evaluate on the test set
    y_pred_rf_no_bootstrap = best_rf_model_no_bootstrap.predict(X_test)
    test_mape_rf_no_bootstrap = mape(y_test, y_pred_rf_no_bootstrap)
    
    print("Test MAPE for Random Forest (No Bootstrap):", test_mape_rf_no_bootstrap)

    return best_rf_model_no_bootstrap, avg_cv_mape_no_bootstrap, test_mape_rf_no_bootstrap


if __name__ == "__main__":
    df_vape = prepare_df()
    X = df_vape[['tax_increase', 'outlet_reduction', 'dec_smoking_prevalence', 
              'dec_tobacco_supply', 'dec_smoking_uptake', 'average_age', 
              'gender_idx', 'ethnicity_idx']]
    y = df_vape[['qalys_pc']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Ensure that y is a 1D array for compatibility
    y_flat = y_train.values.flatten()

    # Generate synthetic samples with fixed random state
    X_res, y_res = generate_synthetic_samples_imbalanced(X_train.values, y_flat, n_samples=200, random_state=42)
    
    # Stack the original and synthetic data
    X_full = np.vstack([X_train.values, X_res])
    y_full = np.hstack([y_flat, y_res])
    
    # Convert to DataFrame for easier handling
    df_resampled_imb = pd.DataFrame(X_full, columns=X.columns)
    df_resampled_imb['qalys_pc'] = y_full

    columns = ['tax_increase', 'outlet_reduction', 'dec_smoking_prevalence', 
              'dec_tobacco_supply', 'dec_smoking_uptake', 'average_age', 
              'gender_idx', 'ethnicity_idx']
    
    X_resampled_imb = df_resampled_imb[columns]
    y_resampled_imb = df_resampled_imb[['qalys_pc']]

    best_rf_model, avg_cv_mape, test_mape_rf = build_model(X_resampled_imb, X_test, y_resampled_imb, y_test, cv=5)
    