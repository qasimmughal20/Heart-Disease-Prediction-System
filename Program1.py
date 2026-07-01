#All importamnt libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix,accuracy_score,
                             classification_report,precision_score,
                             recall_score,f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV


#Dataset -> kaggle heart disease dataset
df = pd.read_csv("heart.csv")
# Remove exact duplicate rows. This dataset (the 1025-row Kaggle copy of heart.csv) contains
# heavy duplication (only ~310 unique patient rows out of 1035). Training/testing on duplicated
# rows leaks identical rows into both splits and produces fake, inflated accuracy.
df = df.drop_duplicates().reset_index(drop=True)
# LABEL FIX: this dataset's target column is encoded backwards vs the standard UCI convention.
# Verified empirically (see project notes): clinically risky values correlate with target=1
# in the wrong direction. Flip so 1 = heart disease present, 0 = no heart disease.
df['target'] = 1 - df['target']
df['target_label'] = df['target'].map({0: 'No Disease', 1: 'Heart Disease'})

df.head()
df.shape
df.columns
df.isnull().sum()
df.info()
df.describe()


#Outliers in dataset
#checing for cholesterol outliers
plt.boxplot(df['chol'])
plt.title("Cholesterol Outliers")
plt.show()

#checking for BP outliers
plt.boxplot(df['trestbps'])
plt.title("Resting BP Outliers")
plt.show()

#checking for Max HR outliers
plt.boxplot(df['thalach'])
plt.title("Max Heart Rate Outliers")
plt.show()


#EDA visualization
#Age distribution
plt.hist(df['age'], bins=20, color='skyblue')
plt.title("Age Distribution of Patients")
plt.xlabel("Age of Patients (Years)")
plt.ylabel("Number of Patients")
plt.show()

#Cholesterol distribution
plt.hist(df['chol'], bins=20, color='orange')
plt.title("Cholesterol Level Distribution")
plt.xlabel("Cholesterol Level (mg/dL)")
plt.ylabel("Number of Patients")
plt.show()

#CP Type Graph
cp_mapping = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}

df['cp_label'] = df['cp'].map(cp_mapping)
sns.countplot(x='cp_label', hue='target_label', data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=30)
plt.show()

#slope
slope_mapping = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping"
}
df['slope_label'] = df['slope'].map(slope_mapping)
sns.countplot(x='slope_label', hue='target_label', data=df)
plt.title("Slope Type vs Heart Disease")
plt.xlabel("Slope Type (ECG)")
plt.ylabel("Number of Patients")
plt.show()

#Max HR 
sns.boxplot(x='target_label', y='thalach', data=df)
plt.xlabel("Heart Disease Status")
plt.title("Max Heart Rate by Target")
plt.show()

#correlationheatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.drop(['target_label','cp_label','slope_label'], axis=1).corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Converts non-numeric features into numeric (0/1) columns
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)


#We need StandardScaler to bring all numeric columns into same scale.
scaler = StandardScaler()
cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


X = df.drop(['target', 'target_label', 'cp_label', 'slope_label'], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Model training (logistic regression and random forest)
# 1st algorthm logistic regression
# Create model
model=LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", lr_accuracy)

# Detailed report
print("Classification Report:", classification_report(y_test, y_pred))


# 2nd algorthm named random forest
# Step 1: Create Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 2: Train the model
rf_model.fit(X_train, y_train)

#feature importance
importance = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Select only important features
important_features = importance_df[importance_df['Importance'] > 0.04]

print("Important Features:")
print(important_features)

# Plot only important features
plt.figure(figsize=(8,6))
plt.barh(important_features['Feature'], important_features['Importance'])
plt.title("Important Features Only")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()


# RANDOM FOREST USING IMPORTANT FEATURES ONLY
# Keep only selected important features
selected_columns = important_features['Feature'].tolist()
X_selected = X[selected_columns]

# Train-test split again
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
rf_selected = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train model
rf_selected.fit(X_train_sel, y_train_sel)
# Predict
y_pred_selected = rf_selected.predict(X_test_sel)
# Accuracy
print("Accuracy with Important Features:")
rf_accuracy = accuracy_score(y_test_sel, y_pred_selected)
print(rf_accuracy)
# Classification Report
print("Classification Report:")
print(classification_report(y_test_sel, y_pred_selected))


# Confusion Matrix
cm_selected = confusion_matrix(y_test_sel, y_pred_selected)
plt.figure(figsize=(5,4))
sns.heatmap(cm_selected, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest (Important Features Only)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Confusion matrix
def plot_conf_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d' , cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(y_test, y_pred, "Logistic Regression Confusion Matrix")
plot_conf_matrix(
    y_test_sel,
    y_pred_selected,
    "Random Forest Confusion Matrix"
)

\
# Accuracy comparison in both models Graph
models = ['Logistic Regression', 'Random Forest']
rf_accuracy = accuracy_score(y_test_sel, y_pred_selected)
accuracies = [lr_accuracy, rf_accuracy]
plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.show()


# precision_score, recall_score, f1_score
# LOGISTIC REGRESSION METRICS GRAPH
print("Logistic Regression Metrics:")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

lr_precision = precision_score(y_test, y_pred)
lr_recall = recall_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)
lr_metrics = [lr_accuracy,lr_precision,lr_recall,lr_f1]
lr_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(7,5))
plt.bar(lr_labels, lr_metrics)
plt.title("Logistic Regression Metrics")
plt.ylabel("Score")
plt.ylim(0,1)
plt.show()

# RANDOM FOREST METRICS GRAPH
print("Random Forest Metrics:")
print("Precision:", precision_score(y_test_sel, y_pred_selected))
print("Recall:", recall_score(y_test_sel, y_pred_selected)) 
print("F1 Score:", f1_score(y_test_sel, y_pred_selected))

rf_precision = precision_score(y_test_sel, y_pred_selected)
rf_recall = recall_score(y_test_sel, y_pred_selected)
rf_f1 = f1_score(y_test_sel, y_pred_selected)
rf_metrics = [rf_accuracy,rf_precision,rf_recall,rf_f1]
rf_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(7,5))
plt.bar(rf_labels, rf_metrics)
plt.title("Random Forest Metrics")
plt.ylabel("Score")
plt.ylim(0,1)
plt.show()

# COMBINED MODEL COMPARISON GRAPH
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lr_values = [lr_accuracy,lr_precision,lr_recall,lr_f1]
rf_values = [rf_accuracy,rf_precision,rf_recall,rf_f1]
x = np.arange(len(metrics_names))
width = 0.35
plt.figure(figsize=(8,5))
plt.bar(x - width/2, lr_values, width,
        label='Logistic Regression')
plt.bar(x + width/2, rf_values, width,
        label='Random Forest')
plt.xticks(x, metrics_names)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.ylim(0,1)
plt.show()


# ROC / AUC CURVE
# Logistic Regression probabilities
y_prob_lr = model.predict_proba(X_test)[:, 1]

# Random Forest probabilities
y_prob_rf = rf_selected.predict_proba(X_test_sel)[:, 1]

# ROC values
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test_sel, y_prob_rf)

# AUC scores
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC Curve
plt.figure(figsize=(7,5))
plt.plot(fpr_lr, tpr_lr,
         label=f'Logistic Regression AUC = {auc_lr:.2f}')
plt.plot(fpr_rf, tpr_rf,
         label=f'Random Forest AUC = {auc_rf:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()



#hyperparameter tuning
params = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))




#T-SNE Graph
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
    
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)
plt.title("t-SNE Visualization of Heart Disease Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


#UMAP 
umap_model = umap.UMAP(random_state=42)
X_umap = umap_model.fit_transform(X)

plt.scatter(X_umap[:,0], X_umap[:,1], c=y)
plt.title("UMAP Visualization of Heart Disease Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
