# Stroke Prediction Project - Milestones 1 & 2
# Course: CSE381 - Introduction to Machine Learning
# Milestone 1: Data Exploration, Cleaning, Naive Bayes & SVM
# Milestone 2: KNN, Decision Tree & Clustering Analysis

# 1. Imports and Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

# 2. Load Dataset
# Ensure the CSV is in your working directory or adjust the path accordingly
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 3. Milestone 1: Data Exploration & Visualization
# 3.1 Descriptive Statistics
print(df.describe(include='all'))

# 3.2 Missing Values
print("Missing values per column:")
print(df.isnull().sum())

# 3.3 Target Distribution
sns.countplot(x='stroke', data=df)
plt.title('Stroke vs Non-Stroke Counts')
plt.show()

# 3.4 Correlation Heatmap (numeric only)
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 3.5 Age vs Glucose vs BMI Scatter
sns.scatterplot(x='age', y='avg_glucose_level', hue='stroke', data=df)
plt.title('Age vs Glucose Level by Stroke')
plt.show()

# 3.6 PCA, LDA, t-SNE for 2D Visualization
# Preprocess for dimension reduction: drop ID and NaNs
df_dr = df.drop('id', axis=1).dropna()
x_dr = pd.get_dummies(df_dr.drop('stroke', axis=1), drop_first=True)
y_dr = df_dr['stroke']

# Standardize
scaler_dr = StandardScaler()
x_dr_scaled = scaler_dr.fit_transform(x_dr)

# PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(x_dr_scaled)
plt.figure()
plt.scatter(pc[:,0], pc[:,1], c=y_dr, cmap='viridis', alpha=0.7)
plt.title('PCA Projection')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()

# LDA
lda = LDA(n_components=1)
lda_proj = lda.fit_transform(x_dr_scaled, y_dr)
plt.figure()
plt.scatter(lda_proj, np.zeros_like(lda_proj), c=y_dr, cmap='coolwarm', alpha=0.7)
plt.title('LDA Projection')
plt.xlabel('LD1')
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(x_dr_scaled)
plt.figure()
plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=y_dr, cmap='rainbow', alpha=0.7)
plt.title('t-SNE Projection')
plt.show()

# 4. Milestone 1: Data Cleaning & Preprocessing
# 4.1 Handle missing BMI by median imputation
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# 4.2 Remove any zero Age or Glucose if biologically invalid
# (if present)
df = df[df['age']>0]
df = df[df['avg_glucose_level']>0]

# 4.3 Encode categorical variables
cat_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4.4 Split features and target
X = df.drop(['id','stroke'], axis=1)
y = df['stroke']

# 4.5 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4.6 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Milestone 1: Training & Testing Classifiers
# 5.1 Naïve Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# 5.2 Support Vector Machine with Grid Search
svc = SVC(probability=True, random_state=42)
param_svc = {'C':[0.1,1,10], 'kernel':['linear','rbf']}
grid_svc = GridSearchCV(svc, param_svc, cv=5, scoring='f1')
grid_svc.fit(X_train_scaled, y_train)
best_svc = grid_svc.best_estimator_
print("Best SVM Params:", grid_svc.best_params_)
y_pred_svm = best_svc.predict(X_test_scaled)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# 6. Milestone 2: KNN & Decision Tree
# 6.1 K-Nearest Neighbors
knn = KNeighborsClassifier()
param_knn = {'n_neighbors': list(range(3,12))}
grid_knn = GridSearchCV(knn, param_knn, cv=5, scoring='f1')
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_
print("Best KNN Params:", grid_knn.best_params_)
y_pred_knn = best_knn.predict(X_test_scaled)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# 6.2 Decision Tree
dt = DecisionTreeClassifier(random_state=42)
param_dt = {'max_depth': list(range(3,15)), 'min_samples_split':[2,5,10]}
grid_dt = GridSearchCV(dt, param_dt, cv=5, scoring='f1')
grid_dt.fit(X_train_scaled, y_train)
best_dt = grid_dt.best_estimator_
print("Best Decision Tree Params:", grid_dt.best_params_)
y_pred_dt = best_dt.predict(X_test_scaled)
print("Decision Tree Report:")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Optional: visualize tree
plt.figure(figsize=(12,8))
plot_tree(best_dt, feature_names=X.columns, class_names=['No Stroke','Stroke'], filled=True)
plt.show()

# 7. Milestone 2: Clustering Analysis
# 7.1 Hierarchical Clustering
linked = linkage(X_train_scaled, method='ward')
plt.figure(figsize=(10,5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# 7.2 K-Means Clustering
# Determine optimal k via elbow method
sse = []
for k in range(2,10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_train_scaled)
    sse.append(km.inertia_)
plt.figure()
plt.plot(range(2,10), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Choose k (e.g., 4)
km_final = KMeans(n_clusters=4, random_state=42)
clusters = km_final.fit_predict(X_train_scaled)

# Analyze clusters vs actual stroke
cluster_df = pd.DataFrame(X_train_scaled, columns=X.columns)
cluster_df['cluster'] = clusters
cluster_df['stroke'] = y_train.values
print(cluster_df.groupby('cluster')['stroke'].mean()) 