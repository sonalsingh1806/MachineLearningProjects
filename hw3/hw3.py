import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# DATA EXPLORATION #

#print out data info
print("Dataset info:")
print(train_data.info())
print("Dataset description:")
print(train_data.describe())

# List out missing values
missing_values = train_data.isna().sum()

#listing the columns with missing values
columns_with_missing = missing_values[missing_values > 0]
print("\nColumns with Missing Values:")
print(columns_with_missing)

# bar plot for target
sns.countplot(x='target', data=train_data)
plt.show()

# correlation heatmap
correlation_matrix = train_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Preprocessing
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Standardize the data (use the same scaler for both train and test data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on train
X_test_scaled = scaler.transform(test_data)  # Transform on test (no fitting)

# Initialize the classifiers
log_reg = LogisticRegression()
naive_bayes = GaussianNB()
#rf = RandomForestClassifier()
rf = RandomForestClassifier(n_estimators=10)
balanced_rf = BalancedRandomForestClassifier()
base_estimator = DecisionTreeClassifier(max_depth=1)
# Base estimator (Decision Tree Classifier with max_depth=1)
base_estimator = DecisionTreeClassifier(max_depth=1)
# AdaBoost with a base estimator (use the model directly)
adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=30)
grad_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,max_depth=3,max_features='sqrt')  # Lower learning rate

# Create the ensemble model with soft voting
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', LogisticRegression()),
        ('naive_bayes', GaussianNB()),
        ('rf', RandomForestClassifier(n_estimators=10))
    ],
    voting='soft'  # Use soft voting to enable predict_proba
)


# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    
    # Predict on the training data (optional, can be used to evaluate overfitting)
    y_train_pred = model.predict(X_train)
    
    # Predict on the test data
    y_test_pred = model.predict(X_test)
    
    # Performance on training data
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    # Output results for training
    print(f"Model: {model.__class__.__name__}")
    print("Training Performance:")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F-Score: {train_f1:.4f}")
    print(f"AUC: {train_auc:.4f}")
    print("-" * 50)
    
    return y_test_pred

# Train and test on the test data
print("Evaluating Logistic Regression...")
log_reg_predictions = evaluate_model(log_reg, X_train_scaled, y_train, X_test_scaled)

print("Evaluating Naïve Bayes...")
naive_bayes_predictions = evaluate_model(naive_bayes, X_train_scaled, y_train, X_test_scaled)

print("Evaluating Random Forest...")
rf_predictions = evaluate_model(rf, X_train_scaled, y_train, X_test_scaled)

# Apply Random Undersampling (RandomUnderSampler)
undersampler = RandomUnderSampler(sampling_strategy='auto')
X_train_under, y_train_under = undersampler.fit_resample(X_train_scaled, y_train)

print("Evaluating Logistic Regression with Random Undersampling...")
log_reg_predictions_under = evaluate_model(log_reg, X_train_under, y_train_under, X_test_scaled)

print("Evaluating Naïve Bayes with Random Undersampling...")
naive_bayes_predictions_under = evaluate_model(naive_bayes, X_train_under, y_train_under, X_test_scaled)

print("Evaluating Random Forest with Random Undersampling...")
rf_predictions_under = evaluate_model(rf, X_train_under, y_train_under, X_test_scaled)

# Apply Random Oversampling
random_oversampler = RandomOverSampler(sampling_strategy='auto')
X_train_resampled, y_train_resampled = random_oversampler.fit_resample(X_train_scaled, y_train)

print("Evaluating Logistic Regression with Random Oversampling...")
log_reg_predictions_resampled = evaluate_model(log_reg, X_train_resampled, y_train_resampled, X_test_scaled)

print("Evaluating Naïve Bayes with Random Oversampling...")
naive_bayes_predictions_resampled = evaluate_model(naive_bayes, X_train_resampled, y_train_resampled, X_test_scaled)

print("Evaluating Random Forest with Random Oversampling...")
rf_predictions_resampled = evaluate_model(rf, X_train_resampled, y_train_resampled, X_test_scaled)

# Apply SMOTE for Over-sampling
smote = SMOTE(sampling_strategy='auto')
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Evaluating Logistic Regression with SMOTE (Oversampling)...")
log_reg_predictions_smote = evaluate_model(log_reg, X_train_smote, y_train_smote, X_test_scaled)

print("Evaluating Naïve Bayes with SMOTE (Oversampling)...")
naive_bayes_predictions_smote = evaluate_model(naive_bayes, X_train_smote, y_train_smote, X_test_scaled)

print("Evaluating Random Forest with SMOTE (Oversampling)...")
rf_predictions_smote = evaluate_model(rf, X_train_smote, y_train_smote, X_test_scaled)

# Apply TomekLinks (Undersampling)
from imblearn.under_sampling import TomekLinks
from sklearn.decomposition import PCA

pca = PCA(n_components=20)  # Reduce to 20 principal components (adjust as necessary)
X_train_scaled_pca = pca.fit_transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

tomek = TomekLinks(sampling_strategy='auto')
X_train_tomek, y_train_tomek = tomek.fit_resample(X_train_scaled_pca, y_train)

print("Evaluating Logistic Regression with TomekLinks (Undersampling)...")
log_reg_predictions_tomek = evaluate_model(log_reg, X_train_tomek, y_train_tomek, X_test_scaled_pca)

print("Evaluating Naïve Bayes with TomekLinks (Undersampling)...")
naive_bayes_predictions_tomek = evaluate_model(naive_bayes, X_train_tomek, y_train_tomek, X_test_scaled_pca)

print("Evaluating Random Forest with TomekLinks (Undersampling)...")
rf_predictions_tomek = evaluate_model(rf, X_train_tomek, y_train_tomek, X_test_scaled_pca)

# Balanced Random Forest (no resampling, just a balanced version of random forest)
print("Evaluating Balanced Random Forest...")
balanced_rf_predictions = evaluate_model(balanced_rf, X_train_scaled, y_train, X_test_scaled)

# AdaBoost with Random Undersampling
print("Evaluating AdaBoost with Random Undersampling...")
adaboost_predictions = evaluate_model(adaboost, X_train_resampled, y_train_resampled, X_test_scaled)

# Gradient Boosting Classifier
print("Evaluating Gradient Boosting Classifier...")
grad_boost_predictions = evaluate_model(grad_boost, X_train_scaled, y_train, X_test_scaled)

# Evaluate Ensemble Model (Logistic Regression, Naïve Bayes, Random Forest)
print("Evaluating Ensemble Model (Logistic Regression, Naïve Bayes, Random Forest)...")
ensemble_predictions = evaluate_model(ensemble_model, X_train_scaled, y_train, X_test_scaled)

# Saving predictions to CSV for future use
predictions = pd.DataFrame({
    "Logistic_Regression": log_reg_predictions,
    "Naive_Bayes": naive_bayes_predictions,
    "Random_Forest": rf_predictions,
    "Logistic_Regression_Undersampled": log_reg_predictions_under,
    "Naive_Bayes_Undersampled": naive_bayes_predictions_under,
    "Random_Forest_Undersampled": rf_predictions_under,
    "Logistic_Regression_Resampled": log_reg_predictions_resampled,
    "Naive_Bayes_Resampled": naive_bayes_predictions_resampled,
    "Random_Forest_Resampled": rf_predictions_resampled,
    "Logistic_Regression_SMOTE": log_reg_predictions_smote,
    "Naive_Bayes_SMOTE": naive_bayes_predictions_smote,
    "Random_Forest_SMOTE": rf_predictions_smote,
    "Logistic_Regression_TomekLinks": log_reg_predictions_tomek,
    "Naive_Bayes_TomekLinks": naive_bayes_predictions_tomek,
    "Random_Forest_TomekLinks": rf_predictions_tomek,
    "Balanced_Random_Forest": balanced_rf_predictions,
    "AdaBoost": adaboost_predictions,
    "Gradient_Boosting": grad_boost_predictions
})

predictions.to_csv("model_predictions.csv", index=False)
print("Predictions saved to 'model_predictions.csv'")