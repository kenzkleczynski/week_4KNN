# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay


# %% # %% 1. Use the question/target variable you submitted

# "Does private vs public status predict whether a student receives high financial aid?"
# Target variable: aid_f (0 = Low aid, 1 = High aid)
# Features: control_Private for-profit, control_Private not-for-profit, and control_Public and others

# COLLEGE DATA SET from functions from last lab
# Load college data from URL
def load_college_data(url):
    college = pd.read_csv(url)
    return college

# Convert columns to right data types
def convert_college_types(df):
    df = df.copy()
    cols = ["level", "control", "hbcu", "flagship"]
    df[cols] = df[cols].astype('category')
    df["vsa_year"] = df["vsa_year"].astype('Int64')
    return df

# One-hot encode categorical variables
def encode_college_features(df):
    df = df.copy()
    cols = ["level", "control", "hbcu", "flagship"]
    df_encoded = pd.get_dummies(df, columns=cols)
    return df_encoded

# Drop unnecessary/incomplete columns
def drop_college_columns(df):
    df = df.copy()
    vsa_cols = ["vsa_year", "vsa_grad_after4_first", 
                "vsa_grad_elsewhere_after4_first", "vsa_enroll_after4_first",
                "vsa_enroll_elsewhere_after4_first", "vsa_grad_after6_first",
                "vsa_grad_elsewhere_after6_first", "vsa_enroll_after6_first",
                "vsa_enroll_elsewhere_after6_first", "vsa_grad_after4_transfer",
                "vsa_grad_elsewhere_after4_transfer", "vsa_enroll_after4_transfer",
                "vsa_enroll_elsewhere_after4_transfer", "vsa_grad_after6_transfer",
                "vsa_grad_elsewhere_after6_transfer", "vsa_enroll_after6_transfer", 
                "vsa_enroll_elsewhere_after6_transfer"]
    
    identifier_cols = ["index", "unitid", "chronname", "long_x", "lat_y",
                       "site", "similar", "nicknames", "counted_pct"]
    
    df = df.drop(vsa_cols + identifier_cols, axis=1)
    return df

# Create binary target variable for financial aid (1 = high aid, 0 = low aid)
def create_college_target(df, threshold=9343):
    df = df.copy()
    df['aid_f'] = pd.cut(df.aid_value, bins=[-1, threshold, df['aid_value'].max()], labels=[0, 1])
    df['aid_f'] = df['aid_f'].astype('Int64')
    prevalence = (df.aid_f.value_counts()[1] / len(df.aid_f))
    print(f"Prevalence: {prevalence:.2%}")
    return df

# Drop features that cause data leakage or have too much missing data
def clean_college_features(df):
    df = df.copy()
    categorical_drops = ['city', 'state', 'basic']
    percentile_drops = ['awards_per_value', 'awards_per_state_value', 
                        'awards_per_natl_value', 'exp_award_value',
                        'exp_award_state_value', 'exp_award_natl_value',
                        'exp_award_percentile', 'fte_percentile',
                        'med_sat_percentile', 'endow_percentile',
                        'grad_100_percentile', 'grad_150_percentile',
                        'pell_percentile', 'retain_percentile',
                        'ft_fac_percentile']
    leakage_drops = ['aid_value', 'aid_percentile']
    missing_drops = ["med_sat_value", "endow_value", 
                     'grad_100_value', 'grad_150_value', 
                     'cohort_size']
    
    all_drops = categorical_drops + percentile_drops + leakage_drops + missing_drops
    df = df.drop(all_drops, axis=1)
    df = df.dropna(subset=["aid_f"])
    return df

# Split data into train, tune, and test sets (60/20/20)
def split_college_data(df, train_size=0.6):
    c_train, c_leftover = train_test_split(df, train_size=train_size, 
                                           stratify=df.aid_f)
    c_tune, c_test = train_test_split(c_leftover, train_size=0.5, 
                                      stratify=c_leftover.aid_f)
    
    print(f"Train prevalence: {(c_train.aid_f.mean() * 100):.2f}%")
    print(f"Tune prevalence: {(c_tune.aid_f.mean() * 100):.2f}%")
    print(f"Test prevalence: {(c_test.aid_f.mean() * 100):.2f}%")
    
    return c_train, c_tune, c_test

# Normalize continuous variables using MinMaxScaler
def normalize_college_data(c_train, c_tune, c_test, target_col='aid_f'):
    c_train = c_train.copy()
    c_tune = c_tune.copy()
    c_test = c_test.copy()
    
    numeric_cols = list(c_train.select_dtypes('number').columns)
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    scaler = MinMaxScaler()
    scaler.fit(c_train[numeric_cols])
    
    c_train[numeric_cols] = scaler.transform(c_train[numeric_cols])
    c_tune[numeric_cols] = scaler.transform(c_tune[numeric_cols])
    c_test[numeric_cols] = scaler.transform(c_test[numeric_cols])
    
    return c_train, c_tune, c_test, scaler

# Complete college pipeline
def college_pipeline(url, threshold=9343, train_size=0.6):
    df = load_college_data(url)
    df = convert_college_types(df)
    df = encode_college_features(df)
    df = drop_college_columns(df)
    df = create_college_target(df, threshold)
    df = clean_college_features(df)
    c_train, c_tune, c_test = split_college_data(df, train_size)
    c_train, c_tune, c_test, scaler = normalize_college_data(c_train, c_tune, c_test)

    return c_train, c_tune, c_test


#%% 
# Run College Pipeline
college_url = "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
c_train, c_tune, c_test = college_pipeline(college_url, threshold=9343)
print("college:", c_train.shape, c_tune.shape, c_test.shape)

# %% 2. Build a kNN model to predict your target variable using 3 nearest neighbors. 
# Make sure it is a classification problem, meaning if needed changed the target variable.

# Drop Nans
c_train_clean = c_train.dropna()
c_tune_clean = c_tune.dropna()
c_test_clean = c_test.dropna()

# Separate features (X) and target (y)
X_train = c_train_clean.drop('aid_f', axis=1)
y_train = c_train_clean['aid_f']
X_tune = c_tune_clean.drop('aid_f', axis=1)
y_tune = c_tune_clean['aid_f']
X_test = c_test_clean.drop('aid_f', axis=1)
y_test = c_test_clean['aid_f']


# Build kNN with k=3
k = 3 
# Create a fitted model instance:
model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
model = model.fit(X_train,y_train) # Fit the model, training set

# %% 3. Create a dataframe that includes the test target values, 
# test predicted values, and test probabilities of the positive class.

y_test_pred = model.predict(X_test) # Predictions, test set
y_test_prob = model.predict_proba(X_test)[:, 1]  # Probabilities of high aid (1's), test set

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Probability_High_Aid': y_test_prob
})

print(results_df.head())



# %% 4. No code question: 
# If you adjusted the k hyperparameter what do you think would happen to the threshold function? 
# Would the confusion matrix look the same at the same threshold levels or not? Why or why not?

# If I increased the k hyperparameter I dont think it would make changes to the threshold function unless 
# I manually edited to not by 50/50 or if i wrote in teh function for it it test the differnt thresholds.
# I think the donfucsion matix would look differed if the threhold levels changes becuase as you increa the k,
# it might begin to get data points that are not that similar to it, but need to find the closests ones. Due to this,
# the threshlad might need to be more picky about what it counts as high aid or not. 


# %% 5. Evaluate the results using the confusion matrix. 
# Then "walk" through your question, summarize what concerns or positive
# elements do you have about the model as it relates to your question?

# create a confusion matrix
plot = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, 
                                             cmap='Blues') #change color to see differences
plot.figure_.suptitle("Confusion Matrix")
plt.show()



# %% 6. Create two functions: 
# One that cleans the data & splits into training|test and 
# one that allows you to train and test the model with different k and threshold values, 
# then use them to optimize your model (test your model with several k and threshold combinations). 
# Try not to use variable names in the functions, but if you need to that's fine. 
# (If you can't get the k function and threshold function to work in one function just run them separately.) 



# %% 7. How well does the model perform?
# Did the interaction of the adjusted thresholds and k values help the model? 
# Why or why not? 


# %% 8. Choose another variable as the target in the dataset and create another 
# kNN model using the two functions you created in step 7. 
