# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

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

def clean_and_split_data(df, target_col='aid_f'):
    
    # remove any rows that have missing/null values
    df_clean = df.dropna()
    
    # separate data into features (X) and target (y)
    # X = all columns EXCEPT the target
    # y = only the target column
    X = df_clean.drop(target_col, axis=1)  # axis=1 means drop a column
    y = df_clean[target_col]
    
    # split the data into training and testing sets (60% train, 40% test)
    # X_train & y_train = data the model learns patterns from
    # X_test & y_test = data we use to evaluate how well the model learned
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y) # ensures both sets have the same proportion of 0's and 1's as original data
    return X_train, X_test, y_train, y_test


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, 
                           k_values=[3], thresholds=[0.5], show_confusion=True):
    results = []     # empty list to store results from each k/threshold combination
    # Outer loop to test each k value (number of neighbors to consider)
    for k in k_values:
        # create a new kNN model with the current k value
        model = KNeighborsClassifier(n_neighbors=k)
        # train the model using training data
        model.fit(X_train, y_train)
        # get probability predictions for test data
        # predict_proba gives probability for each class [prob of 0, prob of 1]
        y_prob = model.predict_proba(X_test)[:, 1] #selects only the probability of class 1 (high aid)
        # Inner loop to test each threshold value (cutoff for classification)
        for threshold in thresholds:
            # convert probabilities to actual predictions (0 or 1) using threshold
            # If probability >= threshold, predict 1 (high aid), otherwise predict 0 (low aid)
            # .astype(int) makes True/False to 1/0
            y_pred = (y_prob >= threshold).astype(int)
            # calculate accuracy by comparing predictions to actual values
            acc = accuracy_score(y_test, y_pred)
            # store the results for this combination in results list
            results.append({
                'k': k,                    # Number of neighbors used
                'threshold': threshold,    # Probability cutoff used
                'accuracy': acc            # Percentage of correct predictions
            })
    # Return results as dataframe
    results_df = pd.DataFrame(results)
    return results_df


# %% Test the functions with different k and threshold values
X_train, X_test, y_train, y_test = clean_and_split_data(c_train_clean)

results = train_and_evaluate_knn(
    X_train, X_test, y_train, y_test,
    k_values=[3,5,9,11,13,15,17,19,21],
    thresholds=[0.3,0.4,0.5,0.6,0.7])
# %% 7. How well does the model perform?
# Did the interaction of the adjusted thresholds and k values help the model? 
# Why or why not? 

original_accuracy = results[(results['k'] == 3) & (results['threshold'] == 0.5)]['accuracy'].values[0]
print(original_accuracy)

# Find the best combination
best_result = results.loc[results['accuracy'].idxmax()]
print(best_result)

#The model did pretty good overall. With the original settings (k=3 and threshold=0.5),
# the accuracy was about 0.9123. By changing k and the threshold, I was able to improve
# the accuracy to about 0.9263 when using k=9 and a threshold of 0.6.

# The interaction between k and the threshold did help the model. Increasing k means the
# model looks at more nearby points when it makes a prediction instead of just a few.
# The threshold controls how confident the model has to be before calling something
# “high aid,” so raising it from 0.5 to 0.6 makes the model a little more picky.

# I think this helped because the data probably does not have an even split between
# high aid and low aid. Changing the threshold helps deal with that imbalance, and using
# k=9 gives the model enough neighbors to make a solid decision without being too
# sensitive to random noise or weird points in the data.


# %% 8. Choose another variable as the target in the dataset and create another 
# kNN model using the two functions you created in step 7. 

# This model is now trying to predict if a school is a flagship school instead of predicting financial aid.
X_train, X_test, y_train, y_test = clean_and_split_data(c_train_clean, 'flagship_X')

results_flagship = train_and_evaluate_knn(
    X_train, X_test, y_train, y_test,
    k_values=[3,5,9],
    thresholds=[0.4,0.5,0.6])
print(results_flagship)
# the model predicts flagship status very well (about 98% accuracy),
# and changing k or the threshold does not make much difference,
# which means the features already separate flagship and non-flagship schools clearly.

