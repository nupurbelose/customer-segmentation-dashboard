import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# **1. Data Generation**

# Set seed for reproducibility
np.random.seed(42)

# Define the number of samples
n = 3000

# Generate synthetic data
data = {
    'Income': np.random.randint(20000, 500000, n),
    'Expenditure': np.random.uniform(0.3, 0.8, n),
    'Savings': np.random.uniform(0.05, 0.5, n),
    'Investments': np.random.uniform(0, 50000, n),
    'Risk_Profile': np.random.choice(['Conservative', 'Moderate', 'Aggressive'], n, p=[0.4, 0.4, 0.2]),
    'Debt_Level': np.random.uniform(0, 100000, n),
    'Age': np.random.randint(18, 70, n),
    'Employment_Status': np.random.choice(['Employed', 'Unemployed', 'Freelancer'], n),
    'Credit_Score': np.random.randint(300, 850, n),
    'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n),
    'Geographic_Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n)
}

# Create DataFrame
data = pd.DataFrame(data)

# Adjust expenditure and savings as a fraction of income
data['Expenditure'] = data['Income'] * data['Expenditure']
data['Savings'] = data['Income'] * data['Savings']

# Save the dataset to a CSV file
data.to_csv('customer_data.csv', index=False)

# Preview the dataset
print("Generated Data:")
print(data.head())

# **2. Preprocessing Data**

# Function to preprocess data
def preprocess_data(data):
    # One-hot encode categorical features
    categorical_columns = ['Risk_Profile', 'Employment_Status', 'Marital_Status', 'Geographic_Location']
    
    # Check which categorical columns exist in the DataFrame
    existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
    
    data = pd.get_dummies(data, columns=existing_categorical_columns, drop_first=True)
    
    # Scale numerical features
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove 'Cluster' column if it exists in the DataFrame
    if 'Cluster' in numerical_columns:
        numerical_columns.remove('Cluster')
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_columns])
    
    return scaled_data

# Apply preprocessing
scaled_data = preprocess_data(data.copy())
print("\nScaled Data:")
print(scaled_data[:5])  # Display first 5 scaled values

# **3. K-Means Clustering and Visualization**

# Select features for clustering
features = data[['Income', 'Expenditure', 'Savings', 'Investments', 'Debt_Level']]
X = features.values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means Clustering with optimal clusters
optimal_clusters = 3  # Change based on the elbow method result
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income', y='Expenditure', hue='Cluster', palette='viridis', s=100)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.legend(title='Cluster')
plt.show()

# Profile each cluster by categorical variables
def cluster_profile(data):
    cluster_profiles = data.groupby('Cluster').agg({
        'Risk_Profile': lambda x: x.mode()[0],
        'Employment_Status': lambda x: x.mode()[0],
        'Marital_Status': lambda x: x.mode()[0],
        'Geographic_Location': lambda x: x.mode()[0]
    }).reset_index()
    
    return cluster_profiles

# Save the clustered dataset
data.to_csv('clustered_customer_data.csv', index=False)


# Display cluster summary - only including numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
cluster_summary = data.groupby('Cluster')[numeric_columns].mean().reset_index()
print("\nCluster Summary:")
print(cluster_summary)
cluster_profiles = cluster_profile(data)
print("\nCluster Profiles:")
print(cluster_profiles)
