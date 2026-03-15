import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('loan_detection.csv')
    print("✅ Dataset loaded successfully.")
    print(f"Original shape of the data: {df.shape}")
except FileNotFoundError:
    print("❌ Error: loan_detection.csv not found. Please make sure the file is in the correct directory.")
    exit()

# --- 1. Clean Column Names ---
# Replace '.' with '_' in column names to make them valid Python identifiers
original_columns = df.columns.tolist()
df.columns = df.columns.str.replace('.', '_', regex=False)
print("\n🔄 Standardizing column names...")
# print("Old columns:", original_columns)
# print("New columns:", df.columns.tolist())


# --- 2. Remove Duplicate Rows ---
num_duplicates_before = df.duplicated().sum()
print(f"\nFound {num_duplicates_before} duplicate rows.")

if num_duplicates_before > 0:
    df.drop_duplicates(inplace=True)
    print("✅ Duplicate rows have been removed.")
    print(f"Shape of the data after removing duplicates: {df.shape}")
else:
    print("👍 No duplicate rows to remove.")


# --- 3. Check for Missing Values ---
missing_values = df.isnull().sum().sum()
print(f"\nFound {missing_values} missing (NaN) values.")

if missing_values == 0:
    print("👍 Your dataset has no missing values.")
# If there were missing values, you could handle them like this:
# df.dropna(inplace=True)  # Option 1: Drop rows with missing values
# df.fillna(0, inplace=True) # Option 2: Fill missing values with 0


# --- Summary and Save ---
print("\n\n--- ✨ Cleaning Complete ✨ ---")
print(f"Final shape of the cleaned data: {df.shape}")

# Save the cleaned dataframe to a new CSV file
cleaned_file_name = 'loan_detection_cleaned.csv'
df.to_csv(cleaned_file_name, index=False)
print(f"✅ Cleaned data saved to '{cleaned_file_name}'")

# Display the first 5 rows of the cleaned data
print("\n--- Head of Cleaned Data ---")
print(df.head())
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('loan_detection.csv')

# Separate features (X) and target (y)
X = df.drop('Loan_Status_label', axis=1)
y = df['Loan_Status_label']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


from sklearn.linear_model import LogisticRegression

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
log_reg.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

print("Random Forest model trained successfully.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Evaluate Logistic Regression ---
y_pred_log_reg = log_reg.predict(X_test)
print("--- Logistic Regression Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log_reg))

# --- Evaluate Random Forest ---
y_pred_rf = random_forest.predict(X_test)
print("\n--- Random Forest Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# --- Confusion Matrix Visualization ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1])
ax[1].set_title('Random Forest Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Get feature importances from the Random Forest model
importances = random_forest.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Most Important Features (Random Forest)')
plt.show()