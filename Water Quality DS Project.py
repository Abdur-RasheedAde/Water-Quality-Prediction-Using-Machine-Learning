#!/usr/bin/env python
# coding: utf-8

# ## Water Quality DS Project
# 
# New notebook

# In[ ]:


from deltalake import DeltaTable, write_deltalake
table_path = 'abfss://Bioinformatics@onelake.dfs.fabric.microsoft.com/Dataset.Lakehouse/Tables/water quality dataset' 
storage_options = {"bearer_token": notebookutils.credentials.getToken('storage'), "use_fabric_endpoint": "true"}
dt = DeltaTable(table_path, storage_options=storage_options)
limited_data = dt.to_pyarrow_dataset().head(1000).to_pandas()
display(limited_data)

# Write data frame to Lakehouse
# write_deltalake(table_path, limited_data, mode='overwrite')

# If the table is too large and might cause an Out of Memory (OOM) error,
# you can try using the code below. However, please note that delta_scan with default lakehouse is currently in preview.
# import duckdb
# display(duckdb.sql("select * from delta_scan('/lakehouse/default/Tables/dbo/bigdeltatable') limit 1000 ").df())


# In[ ]:


#Another method of loading the dataset directly using kaggle API
#Install kagglehub
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install kagglehub


# In[ ]:


import os

# Replace with your actual Kaggle username and key
os.environ['KAGGLE_USERNAME'] = 'adeoyeabdurrasheed'
os.environ['KAGGLE_KEY'] = 'cb7a3f2352464104f3e080d3ad213b97'


# In[ ]:


import kagglehub
import pandas as pd
import os

# 1. Download the dataset and get the path to its directory
dataset_dir = kagglehub.dataset_download('adityakadiwal/water-potability')

# 2. Get the correct filename from the directory
filename = "water_potability.csv"

# 3. Construct the full path to the file using os.path.join
file_path = os.path.join(dataset_dir, filename)

# 4. Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows to verify
print("Successfully loaded DataFrame:")
print(df.head())


# In[ ]:


#display first 9 rows including the header
df.head(10)


# In[ ]:


#Data Processing - Handle missing values by filling them with the mean of the column

df.fillna(df.mean(), inplace=True)
df.head(10)


# In[ ]:


#Save descriptive statistics to a CSV file in the Lakehouse
descriptive_stats = df.describe()
descriptive_stats.to_csv("/lakehouse/default/Files/descriptive_statistics.csv" , index=True)

print("Descriptive statistics saved to Lakehouse")


# In[ ]:


pip install seaborn


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Visualize - Histograms for each feature
df.hist(bins=15, figsize=(15, 10), layout=(4, 3))
plt.suptitle('Histograms of Water Quality Features', fontsize=16)

# Save the figure as PNG with 600 DPI
plt.savefig('/lakehouse/default/Files/water_quality_feature_histograms.png', dpi=600, bbox_inches='tight')

# Show the plots
plt.show()


# In[ ]:


# Visualize - Box Plot for each feature
plt.figure(figsize=(15, 10))
df.boxplot()
plt.title('Boxplots of Water Quality Features')

# Save the figure as PNG with 600 DPI
plt.savefig('/lakehouse/default/Files/boxplots_water_quality.png', dpi=600, bbox_inches='tight')

# Show the plots
plt.show()


# In[ ]:


#Visualize - the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Water Quality Features')

# Save the heatmap as a high-resolution PNG
plt.savefig('/lakehouse/default/Files/correlation_heatmap.png', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()


# In[ ]:


#Visualize - thePairplot with hue based on 'Potability'
pairplot = sns.pairplot(df, hue='Potability')

# Save the pairplot as a high-resolution PNG
plt.savefig('/lakehouse/default/Files/water_quality_pairplot.png', dpi=600, bbox_inches='tight')

# Show the pairplot
plt.show()


# In[ ]:


#Compare two different features - Hardness vs. Conductivity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Hardness', y='Conductivity', hue='Potability', data=df)
plt.title('Scatter Plot of Hardness vs. Conductivity')
plt.show()


# In[ ]:


#Compare two different features - Hardness vs. Turbidity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Hardness', y='Turbidity', hue='Potability', data=df)
plt.title('Scatter Plot of Hardness vs. Turbidity')
plt.show()


# In[ ]:


#Compare two different features - Hardness vs. Turbidity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Hardness', y='Conductivity', hue='Organic_carbon', data=df)
plt.title('Scatter Plot of Hardness vs. Conductivity')
plt.show()


# # Interpretation of the above
# **Histograms**
# 
# Histograms provide a visual representation of the distribution of each feature. They help identify the range, central tendency, and spread of the data. For example, the histogram of Hardness shows a relatively normal distribution, while Solids has a more skewed distribution.
# 
# **Boxplots**
# 
# Boxplots are useful for identifying outliers and understanding the spread of the data. They show the median, quartiles, and potential outliers. For instance, the boxplot of Chloramines indicates that there are no significant outliers in this feature.
# 
# **Correlation Matrix**
# 
# The correlation matrix helps identify relationships between features. A high positive correlation (close to 1) indicates a strong positive relationship, while a high negative correlation (close to -1) indicates a strong negative relationship. For example, Hardness and Conductivity have a moderate positive correlation.
# 
# **Pairplot**
# 
# Pairplots provide a comprehensive view of pairwise relationships between features. They help identify patterns and relationships that might not be apparent from individual plots. The pairplot with hue based on Potability helps visualize how different features relate to water potability.
# 
# **Scatter Plots**
# 
# Scatter plots are useful for visualizing the relationship between two features. They help identify trends and patterns. For example, the scatter plot of Hardness vs. Conductivity shows a positive trend, indicating that higher hardness is associated with higher conductivity.
# 

# ### **Machine Learning on Water Quality Data**

# In[ ]:


#Using the same dataset
df.head(10)


# ### Data Preprocessing
# 
# 1. **Handle Missing Values**: Fill missing values with the mean of the respective column.
# 2. **Split Data**: Split the data into features (X) and target (y).
# 3. **Standardize Features**: Standardize the features using `StandardScaler`.

# In[ ]:


#Import all necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Split the data into features (X) and target (y)
X = df.drop(columns=['Potability'])
y = df['Potability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Train Multiple Models
# 
# 1. **Logistic Regression**
# 2. **Random Forest**
# 3. **Gradient Boosting**
# 4. **Support Vector Machine (SVM)**
# 5. **K-Nearest Neighbors (KNN)**
# 6. **Decision Tree**
# 
# Evaluate each model using accuracy, precision, recall, and F1-score.

# In[ ]:


#Define and Machine Learning Models and Choose the Best
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing sets
test_size = 0.1  # Adjust as needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{report}\n")


# ## Key Visualizations:
# 
# **Confusion Matrix:**
# Shows TP, TN, FP, FN.
# 
# **ROC Curve:**
# Evaluates the trade-off between TPR and FPR.
# 
# **Precision-Recall Curve:**
# Highlights performance for imbalanced data.

# In[ ]:


#@title Model Analysis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

# Find the best model based on accuracy
best_model_name = max(results, key=lambda k: results[k])
best_model = models[best_model_name]

# Generate predictions and classification report
y_pred = best_model.predict(X_test)
print(f"\n{best_model_name} Classification Report:")
print(classification_report(y_test, y_pred))

# 1. Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()


# In[ ]:


# 2. ROC Curve
if hasattr(best_model, "predict_proba"):
    y_scores = best_model.predict_proba(X_test)[:, 1]
else:
    y_scores = best_model.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve - {best_model_name}')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# 4. Feature Importance (for tree-based models)
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
    plt.figure(figsize=(10,8))
    feature_importance = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    plt.barh(X_train.columns[sorted_idx], feature_importance[sorted_idx], align='center')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.show()


# ### Select the Best Model
# 
# Compare the accuracy of each model and select the best-performing model.

# In[ ]:


# Select the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")


# **Calculate and visualize the feature importance based on the best model.**

# In[ ]:


#@title Feature Importance
if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X, columns=df.columns[:-1])  # Exclude the target column 'Potability'

# Check the type of the best model
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
    # For RandomForestClassifier and GradientBoostingClassifier
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
elif isinstance(best_model, LogisticRegression):
    # For LogisticRegression
    feature_importances = best_model.coef_[0]
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
elif isinstance(best_model, (SVC, KNeighborsClassifier, DecisionTreeClassifier)):
    # For other models, use permutation importance
    from sklearn.inspection import permutation_importance
    importances = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = importances.importances_mean
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
else:
    print("Feature importance calculation is not supported for this model type.")
    feature_importance_df = None

# Plot feature importances if available
if feature_importance_df is not None:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances')

    # Save the plot with high resolution
    plt.savefig('feature_importances.png', dpi=600)

    # Show the plot
    plt.show()


# # Using the Best Model For Prediction

# In[ ]:


import pandas as pd
# Load data into pandas DataFrame from "/lakehouse/default/Files/test water quality.csv"
df_test = pd.read_csv("/lakehouse/default/Files/test water quality.csv")
display(df_test)
df_test.head(3)


# In[ ]:


#Process the unknown data
# Handle missing values
df_test.fillna(df_test.mean(), inplace=True)

# Ensure the unknown data has the same columns as the training data
if not set(df_test.columns).issubset(set(X.columns)):
    print("Unknown data columns do not match the training data columns.")
    print("Training data columns:", X.columns)
    print("Unknown data columns:", df_test.columns)
    raise ValueError("Columns mismatch between training and unknown data.")

# Standardize the unknown data using the same scaler
df_test_scaled = scaler.transform(df_test)


# In[ ]:


import pandas as pd  # Ensure pandas is imported

# Convert df_test_scaled to a DataFrame with the correct feature names
df_test_scaled_df = pd.DataFrame(df_test_scaled, columns=best_model.feature_names_in_)

# Make predictions using the DataFrame with feature names
predictions = best_model.predict(df_test_scaled_df)

# Add predictions to the original test DataFrame
df_test['Predicted_Potability'] = predictions

# Display the first few rows
print("First few rows of the unknown data with predictions:")
print(df_test.head())


# In[ ]:


df_test.head(19)


# In[ ]:


#Save the predictions to a CSV file
df_test.to_csv("/lakehouse/default/Files/predictions.csv", index=False)


# In[ ]:




