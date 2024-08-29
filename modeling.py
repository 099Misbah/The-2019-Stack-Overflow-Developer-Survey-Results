#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd

# Load data
data = pd.read_csv('m1_survey_data.csv')

# Check for missing columns
required_columns = ['Gender', 'Country', 'EdLevel', 'MainBranch', 'Employment', 'JobSat', 'CareerSat']
missing_columns = [col for col in required_columns if col not in data.columns]
missing_columns


# In[48]:


# Handle missing values
data = data.dropna(subset=['MainBranch', 'Country', 'Employment', 'JobSat', 'CareerSat'])  # Drop rows with essential missing values

# Convert YearsCode and YearsCodePro to numeric, handling specific string values
data['YearsCode'] = data['YearsCode'].replace('Less than 1 year', 0).replace('More than 50 years', 51).astype(float)
data['YearsCodePro'] = data['YearsCodePro'].replace('Less than 1 year', 0).replace('More than 50 years', 51).astype(float)

# Fill missing values with median
data['ConvertedComp'] = data['ConvertedComp'].fillna(data['ConvertedComp'].median())
data['YearsCode'] = data['YearsCode'].fillna(data['YearsCode'].median())
data['YearsCodePro'] = data['YearsCodePro'].fillna(data['YearsCodePro'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())


# In[49]:


# Convert categorical columns
categorical_columns = ['MainBranch', 'Employment', 'Country', 'Student', 'EdLevel', 'UndergradMajor', 
                       'OrgSize', 'DevType', 'CareerSat', 'JobSat', 'MgrIdiot', 'MgrMoney', 'MgrWant', 
                       'JobSeek', 'LastHireDate', 'LastInt', 'FizzBuzz', 'JobFactors', 'ResumeUpdate', 
                       'CurrencySymbol', 'CurrencyDesc', 'CompFreq', 'SOVisit1st', 'SOVisitFreq', 
                       'SOVisitTo', 'SOFindAnswer', 'SOTimeSaved', 'SOHowMuchTime', 'SOAccount', 'SOPartFreq', 
                       'SOJobs', 'EntTeams', 'SOComm', 'WelcomeChange', 'SONewContent', 'SurveyLength', 
                       'SurveyEase', 'Gender', 'Trans', 'Sexuality', 'Ethnicity', 'Dependents']
categorical_columns = [col for col in categorical_columns if col in data.columns]  # Ensure columns exist
data = pd.get_dummies(data, columns=categorical_columns)


# In[50]:


import pandas as pd

# Load data
data = pd.read_csv('m1_survey_data.csv')


# In[51]:


# Check for missing values
missing_values = data.isnull().sum()

# Display columns with missing values
missing_values = missing_values[missing_values > 0]
print(missing_values)


# In[52]:


# Calculate the total number of missing values
total_missing_values = data.isnull().sum().sum()

print(f'Total missing values in the dataset: {total_missing_values}')


# In[53]:


# Get the total number of rows and columns
total_rows, total_columns = data.shape

print(f'Total number of rows: {total_rows}')
print(f'Total number of columns: {total_columns}')


# In[54]:


# Calculate the total number of missing values per column
missing_values_per_column = data.isnull().sum()

# Display columns with missing values and their counts
missing_columns = missing_values_per_column[missing_values_per_column > 0]
print(missing_columns)

# Total number of missing values
total_missing_values = missing_values_per_column.sum()
print(f'Total missing values in the dataset: {total_missing_values}')

# Total number of rows and columns
total_rows, total_columns = data.shape
print(f'Total number of rows: {total_rows}')
print(f'Total number of columns: {total_columns}')


# In[55]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Calculate the total number of missing values per column
missing_values_per_column = data.isnull().sum()

# Filter columns with missing values
missing_columns = missing_values_per_column[missing_values_per_column > 0]

# Plot the missing values per column
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_columns.index, y=missing_columns.values)
plt.xticks(rotation=90)
plt.title('Number of Missing Values per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.show()


# In[56]:


# Drop columns with more than 20% missing values
threshold = 0.2
data = data[data.columns[data.isnull().mean() < threshold]]
print(f'Remaining columns after dropping: {len(data.columns)}')

# Impute missing values for remaining data
# Fill missing values with the median for numerical columns
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].median(), inplace=True)

# Fill missing values with the mode for categorical columns
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Verify that there are no missing values left
print(f'Total missing values after imputation: {data.isnull().sum().sum()}')

# Display the first few rows of the cleaned dataset
print(data.head())


# In[57]:


# Display the initial number of rows
initial_rows = data.shape[0]
print(f"Initial number of rows: {initial_rows}")

# Check for duplicates
duplicates = data.duplicated()
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")



# In[58]:


# Remove duplicate rows
df_cleaned = data.drop_duplicates()

# Display the number of rows after removing duplicates
remaining_rows = df_cleaned.shape[0]
print(f"Number of rows after removing duplicates: {remaining_rows}")


# In[ ]:





# In[59]:


# Function to detect and count outliers based on IQR
def count_outliers(data):
    outliers_dict = {}
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define the criteria for an outlier
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers for each column
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            outliers = ((data[col] < lower_bound[col]) | (data[col] > upper_bound[col]))
            num_outliers = outliers.sum()
            outliers_dict[col] = num_outliers
    
    return outliers_dict

# Function to remove outliers based on IQR
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define the criteria for an outlier
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame
    df_out = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
    return df_out

# Display the initial number of rows
initial_rows = data.shape[0]
print(f"Initial number of rows: {initial_rows}")

# Count outliers
outliers_count = count_outliers(data)
print("Outliers count per column:")
for col, count in outliers_count.items():
    print(f"{col}: {count} outliers")

# Remove outliers
df_cleaned = remove_outliers(data)

# Display the number of rows after removing outliers
remaining_rows = df_cleaned.shape[0]
print(f"Number of rows after removing outliers: {remaining_rows}")


# In[60]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and it has been loaded correctly
# df = pd.read_csv('path_to_your_dataset.csv')

# Plot the distribution of 'ConvertedComp'
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['ConvertedComp'].dropna(), kde=True, bins=30)
plt.title('Converted Compensation Distribution')
plt.xlabel('Converted Compensation')
plt.ylabel('Frequency')
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Create a histogram of the 'Age' column in the df_cleaned DataFrame
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Age'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y')

# Show the plot
plt.show()


# In[14]:


import matplotlib.pyplot as plt

# Assuming df_cleaned has a 'Country' column and the dataset has been preprocessed accordingly
top_countries = df_cleaned['Country'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 10 Countries of Respondents')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()


# In[15]:


import matplotlib.pyplot as plt

# Assuming df_cleaned has an 'Education Level' column
education_counts = df_cleaned['EdLevel'].value_counts()

plt.figure(figsize=(12, 8))
education_counts.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Education Level Distribution')
plt.xlabel('Count')
plt.ylabel('Education Level')
plt.grid(axis='x')

# Show the plot
plt.tight_layout()
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import pandas as pd


# Calculate the top 5 job challenges
top_challenges = df_cleaned['WorkChallenge'].value_counts().nlargest(5)

# Plot the top 5 job challenges using a horizontal bar chart
plt.figure(figsize=(12, 8))
top_challenges.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 5 Job Challenges')
plt.xlabel('Count')
plt.ylabel('Job Challenge')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[17]:


# Calculate the top 5 work preferences
top_preferences = df_cleaned['WorkLoc'].value_counts().nlargest(5)

# Plot the top 5 work preferences using a horizontal bar chart
plt.figure(figsize=(12, 8))
top_preferences.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Top 5 Work Preferences')
plt.xlabel('Count')
plt.ylabel('Work Preference')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[18]:


# Calculate the Gender
top_preferences = df_cleaned['Gender'].value_counts().nlargest(5)

# Plot the top 5 work preferences using a horizontal bar chart
plt.figure(figsize=(12, 8))
top_preferences.sort_values().plot(kind='barh', color='skyblue')
plt.title('Gender')
plt.xlabel('Count')
plt.ylabel('Gender')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[19]:


# Count the frequency of open-source contributions
contribution_freq = df_cleaned['OpenSourcer'].value_counts()

# Plot the frequency of open-source contributions
plt.figure(figsize=(12, 8))
sns.barplot(y=contribution_freq.index, x=contribution_freq.values, palette='viridis')
plt.title('Frequency of Open-Source Contributions')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[20]:


# Count the frequency of opinions on open-source software quality
quality_freq = df_cleaned['OpenSource'].value_counts()

# Plot the frequency of open-source software quality perceptions
plt.figure(figsize=(12, 8))
sns.barplot(y=quality_freq.index, x=quality_freq.values, palette='viridis')
plt.title('Perception of Open-Source Software Quality')
plt.xlabel('Count')
plt.ylabel('Perception')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:





# In[62]:


import pandas as pd


# List of columns to remove
columns_to_remove = [
    'Respondent', 'CurrencySymbol', 'CurrencyDesc', 'CompFreq', 'WorkPlan',
    'CodeRev', 'CodeRevHrs', 'PurchaseHow', 'PurchaseWhat', 'DevEnviron', 'Containers',
    'Blockchain0rg', 'Blockchains', 'BetterLife', 'ITperson', 'Offon', 'SocialMedia',
    'Extraversion', 'ScreenName', 'SOVisit1st', 'SOVisitFreq', 'SOVisitTo', 'SOFindAnswer',
    'SOTimeSaved', 'SOHowMuchTime', 'SOAccount', 'SOPartFreq', 'SOJobs', 'EntTeams', 'SOComm',
    'WelcomeChange', 'SONewContent', 'SurveyLength', 'SurveyEase'
]

# Drop the columns from the DataFrame
data_cleaned = df_cleaned.drop(columns=columns_to_remove, errors='ignore')


print(data_cleaned.head())  # Display the first few rows of the cleaned DataFrame


# In[63]:


# Numerical summary
numerical_summary = data_cleaned.describe()

# Categorical summary
categorical_summary = data_cleaned.describe(include=['object'])

# Missing values summary
missing_values_summary = data_cleaned.isnull().sum()

# Display the summaries
print("Numerical Summary:")
print(numerical_summary)
print("\nCategorical Summary:")
print(categorical_summary)


# In[64]:


# Total number of rows and columns
total_rows, total_columns = data_cleaned.shape
print(f'Total number of rows: {total_rows}')
print(f'Total number of columns: {total_columns}')


# ### Modeling
# ### goal 3) Compensation and Emp Trends

# Here are some potential features that might influence compensation :
# 
# Country
# YearsCode
# YearsCodePro
# EdLevel
# UndergradMajor
# Employment
# DevType
# OrgSize

# ## linear regression model for compensation and emp trends

# In[72]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Select relevant features and the target variable
selected_features = [
    'Country', 'YearsCode', 'YearsCodePro', 'EdLevel', 
    'UndergradMajor', 'Employment', 'DevType', 'OrgSize'
]
X = data_cleaned[selected_features]
y = data_cleaned['ConvertedComp']


# In[66]:


from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame after preprocessing
X = data_cleaned[selected_features]  # Features selected in Step 2
y = data_cleaned['ConvertedComp']    # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## regression model to predict salaries based on relevant features.
# 
# 

# In[75]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Define preprocessor for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), 
         ['YearsCode', 'YearsCodePro']),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), 
         ['Country', 'OrgSize'])
    ])

# Create and train the Gradient Boosting Regressor model
gb_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])

# Train the model
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting Regressor:")
print(f"  Mean Squared Error: {mse_gb:.2f}")
print(f"  R-squared: {r2_gb:.2f}")


# In[73]:


# Display column names
print(data_cleaned.columns.tolist())


# In[ ]:




