from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_curve, auc, confusion_matrix, 
                           classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import export_text, plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import logging
import warnings
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def create_holdout_split(X, y, test_size=0.2, holdout_size=0.1):
    # First split: separate holdout set
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y,
        test_size=holdout_size,
        random_state=42,
        stratify=y
    )
    
    # Second split: create training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=test_size,
        random_state=42,
        stratify=y_temp
    )
    
    logging.info(f"Dataset splits:")
    logging.info(f"Training set size: {X_train.shape[0]}")
    logging.info(f"Validation set size: {X_val.shape[0]}")
    logging.info(f"Holdout set size: {X_holdout.shape[0]}")
    
    return X_train, X_val, X_holdout, y_train, y_val, y_holdout

def evaluate_model_on_set(model, X, y, set_name=""):
    """
    Evaluate model performance on a given dataset.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        set_name: Name of the dataset (e.g., "Training", "Validation", "Holdout")
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Log results
    logging.info(f"\n{set_name} Set Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{set_name} Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{set_name.lower()}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    logging.info("Loading dataset...")
    
    # Load data
    df = pd.read_csv(file_path)
    logging.info(f"Dataset shape: {df.shape}")
    
    # Check class distribution
    class_dist = df.iloc[:, -1].value_counts()
    logging.info(f"Class distribution:\n{class_dist}")
    
    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.replace({'BENIGN': 0, 'DDoS': 1})
    
    # Handle missing and infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_vals = X.isnull().sum()
    logging.info(f"Features with missing values:\n{missing_vals[missing_vals > 0]}")
    X.fillna(X.mean(), inplace=True)
    
    return X, y

def apply_variance_thresholding(X, threshold=0.01):
    """Apply variance thresholding and log the dropped features."""
    logging.info(f"\nApplying variance thresholding with threshold {threshold}")
    logging.info(f"Original number of features: {X.shape[1]}")
    
    # Calculate variances
    variances = X.var()
    logging.info("\nFeature variances:")
    for feature, var in variances.items():
        logging.info(f"{feature}: {var}")
    
    # Apply variance thresholding
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    
    # Get dropped features
    kept_features = X.columns[selector.get_support()].tolist()
    dropped_features = X.columns[~selector.get_support()].tolist()
    
    logging.info(f"\nFeatures after variance thresholding: {X_reduced.shape[1]}")
    logging.info(f"Dropped {len(dropped_features)} features:")
    for feature in dropped_features:
        logging.info(f"- {feature}")
    
    return X_reduced, selector, kept_features

def apply_feature_selection(X, y, feature_names):
    """Apply feature selection using RandomForest importance."""
    logging.info(f"\nApplying feature selection")
    logging.info(f"Features before selection: {X.shape[1]}")
    
    # Initialize and fit selector
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    selector = SelectFromModel(rf, threshold='mean')
    X_reduced = selector.fit_transform(X, y)
    
    # Get selected features
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.get_support()[i]]
    
    logging.info(f"Features after selection: {X_reduced.shape[1]}")
    logging.info("Selected features:")
    for feature in selected_features:
        logging.info(f"- {feature}")
    
    return X_reduced, selector, selected_features

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load data and preprocess
    logging.info("Loading dataset...")
    df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    
    # Log initial data statistics
    logging.info(f"Initial dataset shape: {df.shape}")
    logging.info("\nClass distribution:")
    logging.info(df.iloc[:, -1].value_counts())
    
    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.replace({'BENIGN': 0, 'DDoS': 1})
    
    # Store feature names before preprocessing
    feature_names = X.columns.tolist()
    
    # Handle infinite and missing values
    logging.info("\nHandling infinite and missing values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Log columns with infinite or missing values
    inf_cols = X.columns[~np.isfinite(X).all()].tolist()
    missing_cols = X.columns[X.isna().any()].tolist()
    
    if inf_cols:
        logging.info("Columns with infinite values:")
        for col in inf_cols:
            logging.info(f"- {col}")
            
    if missing_cols:
        logging.info("Columns with missing values:")
        for col in missing_cols:
            logging.info(f"- {col}")
    
    # Create three-way split with stratification
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = create_holdout_split(
        X, y, test_size=0.2, holdout_size=0.15
    )
    
    # Create pipeline with proper preprocessing
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler()),  # Scale the features
        ('variance_threshold', VarianceThreshold(threshold=0.02)),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
        ('classifier', RandomForestClassifier(
            random_state=42,
            oob_score=True,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Rest of the hyperparameter search space remains the same
    param_dist = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [10, 20, 30, 50],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Perform hyperparameter search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        refit='f1',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    
    # Fit on training data
    logging.info("\nTraining model...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # Log best parameters
    logging.info("\nBest parameters found:")
    logging.info(random_search.best_params_)
    
    # Evaluate on all sets unconditionally
    logging.info("\nEvaluating model performance on all sets:")
    
    # Training set evaluation
    train_metrics = evaluate_model_on_set(best_model, X_train, y_train, "Training")
    
    # Validation set evaluation
    val_metrics = evaluate_model_on_set(best_model, X_val, y_val, "Validation")
    
    # Holdout set evaluation
    holdout_metrics = evaluate_model_on_set(best_model, X_holdout, y_holdout, "Holdout")
    
    # Print comparison of F1 scores
    logging.info("\nF1 Score Comparison:")
    logging.info(f"Training F1 Score: {train_metrics['f1']:.4f}")
    logging.info(f"Validation F1 Score: {val_metrics['f1']:.4f}")
    logging.info(f"Holdout F1 Score: {holdout_metrics['f1']:.4f}")
    
    # Get feature importance (only if needed)
    try:
        top_features = analyze_feature_importance(best_model, feature_names)
    except:
        logging.warning("Could not analyze feature importance")
        top_features = None
    
    # Save the final model and metadata (only save what we have)
    model_metadata = {
        'model': best_model,
        'feature_names': feature_names,
        'training_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'holdout_metrics': holdout_metrics,
        'top_features': top_features,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(model_metadata, 'model_with_metadata.joblib')
    logging.info("\nModel and metadata saved as 'model_with_metadata.joblib'")

def analyze_feature_importance(model, feature_names):
    """Analyze and log feature importance."""
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        importances = model.named_steps['classifier'].feature_importances_
        feature_selector = model.named_steps['feature_selection']
        selected_features_mask = feature_selector.get_support()
        
        # Get selected feature names
        selected_features = [f for f, m in zip(feature_names, selected_features_mask) if m]
        
        # Create feature importance pairs
        feature_importance = list(zip(selected_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logging.info("\nTop 10 most important features:")
        for feature, importance in feature_importance[:10]:
            logging.info(f"{feature}: {importance:.4f}")
            
        return feature_importance
    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()