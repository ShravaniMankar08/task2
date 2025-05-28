import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("Titanicdataset.csv")  

#Summary Statistics
print(df.head())
print(df.info())
print(df.describe(include='all'))
#Check for missing values
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histograms
df[numeric_cols].hist(bins=15, figsize=(15, 10), color='skyblue')
plt.suptitle('Histograms of Numeric Features')
plt.show()

# Boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot 
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived', palette='husl')
plt.show()

# Count plots
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in ['Sex', 'Pclass', 'Embarked']:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='Survived')
    plt.title(f'{col} vs Survived')
    plt.show()

# Interactive histogram
fig = px.histogram(df, x='Age', color='Survived', nbins=30, title='Age Distribution by Survival')
fig.show()

# Interactive scatter
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare by Survival')
fig.show()

