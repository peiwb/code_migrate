#!/usr/bin/env python3
"""
Test file for pandas/sklearn data analysis operations
This file contains common data analysis patterns that would need migration
from Python/PySpark to Snowflake/SageMaker
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 1000

    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'years_employed': np.random.exponential(5, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
        'num_accounts': np.random.poisson(3, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'has_mortgage': np.random.choice([True, False], n_samples),
        'purchase_amount': np.random.lognormal(6, 1.5, n_samples)
    }

    # Introduce some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    for idx in missing_indices[:len(missing_indices) // 3]:
        data['income'][idx] = np.nan
    for idx in missing_indices[len(missing_indices) // 3:2 * len(missing_indices) // 3]:
        data['credit_score'][idx] = np.nan
    for idx in missing_indices[2 * len(missing_indices) // 3:]:
        data['years_employed'][idx] = np.nan

    return pd.DataFrame(data)


def basic_data_exploration(df):
    """Perform basic data exploration"""
    print("=== BASIC DATA EXPLORATION ===")

    # Dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values analysis
    print("\nMissing values:")
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    print(missing_summary)

    # Basic statistics for numeric columns
    print("\nNumeric columns summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())

    # Categorical columns analysis
    print("\nCategorical columns:")
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        print(f"\n{col} - Unique values: {df[col].nunique()}")
        print(df[col].value_counts().head())


def data_cleaning_preprocessing(df):
    """Perform data cleaning and preprocessing operations"""
    print("\n=== DATA CLEANING & PREPROCESSING ===")

    df_processed = df.copy()

    # Handle missing values for numeric columns using pandas methods
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

    # Fill missing values with median
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            median_value = df_processed[col].median()
            df_processed[col].fillna(median_value, inplace=True)
            print(f"Filled missing values in {col} with median: {median_value:.2f}")

    # Handle outliers using IQR method for key columns
    outlier_cols = ['income', 'credit_score', 'purchase_amount']
    for col in outlier_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_before = len(df_processed[(df_processed[col] < lower_bound) |
                                           (df_processed[col] > upper_bound)])

        # Cap outliers instead of removing them
        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)

        print(f"Outliers capped in {col}: {outliers_before}")

    # Create derived features
    df_processed['debt_to_income_ratio'] = df_processed['debt_ratio'] * df_processed['income']
    df_processed['credit_score_category'] = pd.cut(df_processed['credit_score'],
                                                   bins=[0, 580, 669, 739, 799, 850],
                                                   labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    df_processed['age_group'] = pd.cut(df_processed['age'],
                                       bins=[0, 25, 35, 50, 65, 100],
                                       labels=['18-25', '26-35', '36-50', '51-65', '65+'])

    # Log transformation for skewed variables
    skewed_cols = ['purchase_amount', 'income']
    for col in skewed_cols:
        df_processed[f'{col}_log'] = np.log1p(df_processed[col])

    print("Data cleaning completed.")
    print(f"Original shape: {df.shape}, Processed shape: {df_processed.shape}")

    return df_processed


def feature_engineering(df):
    """Perform advanced feature engineering"""
    print("\n=== FEATURE ENGINEERING ===")

    df_features = df.copy()

    # Encode categorical variables
    categorical_cols = ['region', 'education', 'marital_status']

    # Label encoding for ordinal variables
    education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_features['education_encoded'] = df_features['education'].map(education_mapping)

    # One-hot encoding for nominal variables
    nominal_cols = ['region', 'marital_status']
    for col in nominal_cols:
        dummies = pd.get_dummies(df_features[col], prefix=col)
        df_features = pd.concat([df_features, dummies], axis=1)

    # Create interaction features
    df_features['income_credit_interaction'] = (df_features['income'] *
                                                df_features['credit_score'])
    df_features['age_income_interaction'] = (df_features['age'] *
                                             df_features['income'])

    # Binning continuous variables
    df_features['income_quartile'] = pd.qcut(df_features['income'],
                                             q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Boolean feature engineering
    df_features['high_earner'] = (df_features['income'] >
                                  df_features['income'].quantile(0.8)).astype(int)
    df_features['excellent_credit'] = (df_features['credit_score'] > 740).astype(int)

    print("Feature engineering completed.")
    print(f"Number of features created: {df_features.shape[1] - df.shape[1]}")

    return df_features


def statistical_analysis(df):
    """Perform statistical analysis"""
    print("\n=== STATISTICAL ANALYSIS ===")

    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()

    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })

    print("Highly correlated feature pairs (|correlation| > 0.7):")
    for pair in high_corr_pairs:
        print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")

    # Group-based analysis
    print("\nGroup-based analysis:")

    # Analyze by categorical variables
    print("\nAverage purchase amount by region:")
    region_analysis = df.groupby('region')['purchase_amount'].agg(['mean', 'median', 'std'])
    print(region_analysis)

    print("\nAverage income by education level:")
    education_analysis = df.groupby('education')['income'].agg(['mean', 'count'])
    print(education_analysis)

    # Pivot table analysis
    print("\nPivot table - Average credit score by education and region:")
    pivot_table = pd.pivot_table(df,
                                 values='credit_score',
                                 index='education',
                                 columns='region',
                                 aggfunc='mean')
    print(pivot_table)


def feature_selection_and_scaling(df, target_col='purchase_amount'):
    """Perform feature selection and scaling"""
    print("\n=== FEATURE SELECTION & SCALING ===")

    # Prepare features for analysis
    # Select only numeric features for this example
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove('customer_id')  # Remove ID column
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    X = df[feature_cols]

    # Create a binary target for classification-based feature selection
    y_binary = (df[target_col] > df[target_col].median()).astype(int)

    # Handle any remaining NaN values using pandas
    X = X.fillna(X.median())

    print(f"Number of features before selection: {X.shape[1]}")

    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y_binary)

    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    feature_scores = selector.scores_

    print(f"Number of features after selection: {X_selected.shape[1]}")
    print("\nTop selected features and their scores:")
    feature_importance = list(zip(selected_features,
                                  [feature_scores[i] for i in selector.get_support(indices=True)]))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, score in feature_importance:
        print(f"{feature}: {score:.2f}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    print(f"\nFeature scaling completed. Mean: {X_scaled.mean():.6f}, Std: {X_scaled.std():.6f}")

    # PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA completed. Components: {X_pca.shape[1]}, Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    return X_scaled, X_pca, selected_features, scaler


def advanced_aggregations(df):
    """Perform advanced aggregation operations"""
    print("\n=== ADVANCED AGGREGATIONS ===")

    # Multiple aggregations
    agg_functions = {
        'income': ['mean', 'median', 'std', 'min', 'max'],
        'credit_score': ['mean', 'std', 'count'],
        'purchase_amount': ['sum', 'mean', 'median'],
        'age': ['mean', 'min', 'max']
    }

    print("Multi-column aggregations by region:")
    region_aggs = df.groupby('region').agg(agg_functions)
    print(region_aggs.round(2))

    # Rolling window analysis (simulate time series)
    df_sorted = df.sort_values('customer_id')
    df_sorted['income_rolling_mean'] = df_sorted['income'].rolling(window=50).mean()
    df_sorted['purchase_rolling_sum'] = df_sorted['purchase_amount'].rolling(window=30).sum()

    print(f"\nRolling statistics computed for {len(df_sorted)} records")

    # Percentile calculations
    print("\nPercentile analysis for key metrics:")
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for col in ['income', 'credit_score', 'purchase_amount']:
        print(f"\n{col} percentiles:")
        for p in percentiles:
            print(f"  {p * 100:5.1f}%: {df[col].quantile(p):10.2f}")


def main():
    """Main function to run all analysis steps"""
    print("Starting pandas/sklearn data analysis test...")

    # Create sample dataset
    df = create_sample_dataset()

    # Run analysis steps
    basic_data_exploration(df)

    df_processed = data_cleaning_preprocessing(df)

    df_with_features = feature_engineering(df_processed)

    statistical_analysis(df_with_features)

    X_scaled, X_pca, selected_features, scaler = feature_selection_and_scaling(df_with_features)

    advanced_aggregations(df_with_features)

    # Summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"After feature engineering: {df_with_features.shape[1]} columns")
    print(f"Selected features for modeling: {len(selected_features)}")
    print(f"PCA components: {X_pca.shape[1]}")
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()