import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Load the dataset
df = pd.read_csv("NC_cleaned.csv")  # Replace with the actual file path

# List out missing values
missing_values = df.isna().sum()

#listing the columns with missing values
columns_with_missing = missing_values[missing_values > 0]
print("\nColumns with Missing Values:")
print(columns_with_missing)


# Display basic info
print("Dataset info:")
print(df.info())

# Data Cleaning Steps

# 1. Drop rows with missing values in critical columns
df = df.dropna(subset=['driver_race', 'driver_age', 'driver_gender', 'is_arrested', 'search_conducted', 'is_arrested', 'stop_outcome'])

# 2. Filter the dataset to include only the required races
valid_races = ['Asian', 'Black', 'Hispanic', 'White','Other']
df = df[df['driver_race'].isin(valid_races)]

# 3. Bucket the driver_age into specified age groups
bins = [0, 19, 29, 39, 49, float('inf')]
labels = ['15-19', '20-29', '30-39', '40-49', '50+']
df['age_group'] = pd.cut(df['driver_age'], bins=bins, labels=labels, right=True)

# 4. Filter out rows where age_group is NaN (e.g., driver_age < 15)
df = df.dropna(subset=['age_group'])

# 5. Ensure is_arrested is boolean (if not already)
df['is_arrested'] = df['is_arrested'].astype(bool)

# Plot distribution of age groups
plt.figure(figsize=(8, 5))
sns.countplot(x='age_group', data=df, palette='viridis')
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Plot distribution of driver gender
plt.figure(figsize=(8, 5))
sns.countplot(x='driver_gender', data=df, palette='Blues')
plt.title('Distribution of Driver Gender (M = Male)')
plt.xlabel('Gender (M = Male)')
plt.ylabel('Count')
plt.show()

# Arrest rate by race
plt.figure(figsize=(10, 6))
sns.barplot(x='driver_race', y='is_arrested', data=df, ci=None, palette='coolwarm')
plt.title('Arrest Rate by Race')
plt.xlabel('Race')
plt.ylabel('Arrest Rate')
plt.show()


# Create a binary target for Citation (1 = Citation, 0 = Other outcomes)
df['citation'] = (df['stop_outcome'] == 'Citation').astype(int)

# Calculate Arrest Percentages

# Group by race, age_group, and gender
grouped = df.groupby(['driver_race', 'age_group', 'driver_gender'])

# Calculate the percentage of arrests
arrest_percentages = grouped['is_arrested'].mean() * 100

# Convert to a DataFrame for better readability
arrest_percentages_df = arrest_percentages.reset_index()
arrest_percentages_df.rename(columns={'is_arrested': 'arrest_percentage'}, inplace=True)

# Display the results
print("\nArrest percentages by race, age group, and gender:")
print(arrest_percentages_df)

# Save the results to a CSV file (optional)
arrest_percentages_df.to_csv("arrest_percentages.csv", index=False)
print("\nResults saved to 'arrest_percentages.csv'.")

# Plot arrest percentages by race, age group, and gender
plt.figure(figsize=(12, 8))
sns.barplot(x='arrest_percentage', y='driver_race', hue='age_group', data=arrest_percentages_df, ci=None, palette='coolwarm')
plt.title('Arrest Percentage by Race and Age Group')
plt.xlabel('Arrest Percentage')
plt.ylabel('Driver Race')
plt.legend(title='Age Group', loc='upper right')
plt.show()

# One-hot encode race and gender
df = pd.get_dummies(df, columns=['driver_race', 'driver_gender'], drop_first=True)

# Check the columns after one-hot encoding
print("Columns after one-hot encoding:")
print(df.columns)

# Ordinal encode age groups
age_mapping = {'15-19': 1, '20-29': 2, '30-39': 3, '40-49': 4, '50+': 5}
df['age_group'] = df['age_group'].map(age_mapping)

# Ensure that all columns are numeric before using them
X = df[['age_group', 'driver_race_Black', 'driver_race_Hispanic', 'driver_race_White', 'driver_gender_M']]

# Ensure X is of numeric type (just in case)
X = X.apply(pd.to_numeric, errors='coerce')


# Remove any rows with NaN values (result of coercion from non-numeric values)
X = X.dropna()


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sampling the data to reduce computational cost (ensure standard errors are < 0.1)
y_search = df['search_conducted']
y_arrest = df['is_arrested']
y_citation = df['citation']

X_sample, _, y_search_sample, _ = train_test_split(X, y_search, test_size=0.8, random_state=42)
_, _, y_arrest_sample, _ = train_test_split(X, y_arrest, test_size=0.8, random_state=42)
_, _, y_citation_sample, _ = train_test_split(X, y_citation, test_size=0.8, random_state=42)




print("Search Conducted Class Distribution:")
print(y_search_sample.value_counts())

print("Arrest Class Distribution:")
print(y_arrest_sample.value_counts())

print("Citation Class Distribution:")
print(y_citation_sample.value_counts())

# Train logistic regression models
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, penalty='l2')  # Increase max_iter if convergence is an issue
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Train for Search Conducted
model_search, mse_search = train_logistic_regression(X_sample, y_search_sample, X_sample, y_search_sample)

# Train for Arrest
model_arrest, mse_arrest = train_logistic_regression(X_sample, y_arrest_sample, X_sample, y_arrest_sample)

# Train for Citation
model_citation, mse_citation = train_logistic_regression(X_sample, y_citation_sample, X_sample, y_citation_sample)

# Report coefficients and MSEs
def report_results(model, mse, feature_names):
    coefficients = model.coef_[0]
    results = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    print("Coefficients:")
    print(results)
    print(f"Mean Squared Error: {mse:.4f}")

# Search Conducted
print("\nSearch Conducted:")
report_results(model_search, mse_search, X.columns)

# Arrest
print("\nArrest:")
report_results(model_arrest, mse_arrest, X.columns)

# Citation
print("\nCitation:")
report_results(model_citation, mse_citation, X.columns)

# Visualize the coefficients
def plot_coefficients(model, feature_names, title):
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
    plt.title(title)
    plt.show()

# Search Conducted Coefficients
plot_coefficients(model_search, X.columns, 'Search Conducted: Coefficients')

# Arrest Coefficients
plot_coefficients(model_arrest, X.columns, 'Arrest: Coefficients')

# Citation Coefficients
plot_coefficients(model_citation, X.columns, 'Citation: Coefficients')




