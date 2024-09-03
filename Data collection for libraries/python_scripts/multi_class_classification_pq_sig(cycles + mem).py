import numpy as np
import pandas as pd
import os

from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def merge_cycle_mem(file_list_1, file_list_2):
    merged_algo_df = []
    for file1 in file_list_1:
        algo_name1 = os.path.basename(file1).split('.out')[0]
        for file2 in file_list_2:
            algo_name2 = os.path.basename(file2).split('.out')[0]
            if algo_name1 == algo_name2:
                # print(file1, file2)
                df1 = pd.read_csv(input_dir_cycles + '/' + file1)
                df2 = pd.read_csv(input_dir_mem + '/' + file2)
                merged_df = pd.concat([df1, df2], axis=1)

                merged_algo_df.append(merged_df)
    return merged_algo_df


#input_dir_cycles = '/home/heresy/Downloads/MLClassification/csv/SIG_perf/'
#input_dir_mem = '/home/heresy/Downloads/MLClassification/csv/SIG_mem'

input_dir_cycles = '/home/heresy/Downloads/CSV/SIG_perf/'
input_dir_mem = '/home/heresy/Downloads/CSV/SIG_mem'

dilithium_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                              file.startswith('Dilithium')]
dilithium_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                           file.startswith('Dilithium')]
merged_dilithium_df = merge_cycle_mem(dilithium_file_path_cycles, dilithium_file_path_mem)

falcon_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                           file.startswith('Falcon')]
falcon_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                        file.startswith('Falcon')]
merged_falcon_df = merge_cycle_mem(falcon_file_path_cycles, falcon_file_path_mem)

sphincs_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                            file.startswith('SPHINCS')]
sphincs_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                         file.startswith('SPHINCS')]
merged_sphincs_df = merge_cycle_mem(sphincs_file_path_cycles, sphincs_file_path_mem)

dsa_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                        file.startswith('ML')]
dsa_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                     file.startswith('ML')]
merged_dsa_df = merge_cycle_mem(dsa_file_path_cycles, dsa_file_path_mem)


file_lengths = {
    'dilithium_files': len(merged_dilithium_df),
    'falcon_files': len(merged_falcon_df),
    'sphincs_files': len(merged_sphincs_df),
    'dsa_files': len(merged_dsa_df)
}

minimal = min(file_lengths.values())

dilithium_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_dilithium_df)) for df in merged_dilithium_df])
falcon_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_falcon_df)) for df in merged_falcon_df])
sphincs_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_sphincs_df)) for df in merged_sphincs_df])
dsa_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_dsa_df)) for df in merged_dsa_df])

final_dataset = pd.concat([dilithium_samples, falcon_samples, sphincs_samples, dsa_samples])
final_dataset = final_dataset.drop(['elapsed_time'], axis=1)
final_dataset.fillna(0, inplace=True)
patterns = [
    (r'^[dD]ilithium.*', 'Dilithium'),
    (r'^Falcon.*', 'Falcon'),
    (r'^ML-DSA.*', 'ML-DSA'),
    (r'^SPHINCS.*', 'SPHINCS')
]
for pattern, replacement in patterns:
    final_dataset['label'] = final_dataset['label'].str.replace(pattern, replacement, regex=True)

scaler = MinMaxScaler()
label_column = 'label'  # Replace with the name of your label column if different
columns_to_scale = [col for col in final_dataset.columns if col != label_column]

label_mapping = {label: idx for idx, label in enumerate(final_dataset['label'].unique())}
final_dataset['label'] = final_dataset['label'].map(label_mapping)
final_dataset[columns_to_scale] = scaler.fit_transform(final_dataset[columns_to_scale])
final_dataset = final_dataset.sample(frac=1)

for label, idx in label_mapping.items():
    print(f"{label}: {idx}")
label_counts = final_dataset['label'].value_counts()
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

output_file = './output.csv'
final_dataset.to_csv(output_file, index=False)

#print(final_dataset.label.value_counts())

X = final_dataset[columns_to_scale]
#X = final_dataset[['CPU5_cycles', 'CPU11_cycles', 'VmSize', 'VmRSS', 'VmData', 'CPU1_cycles', 'CPU8_cycles', 'CPU10_cycles', 'CPU3_cycles', 'CPU0_cycles', 'CPU9_cycles']]
#X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'CPU0_cycles', 'CPU9_cycles', 'CPU1_cycles', 'CPU6_cycles', 'CPU11_cycles', 'VmPTE', 'CPU10_cycles', 'CPU3_cycles']]
#X = final_dataset[['VmSize', 'VmRSS']]
y = final_dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Feature Selection'''


'''
select_k_best = SelectKBest(score_func=chi2, k='all')
select_k_best.fit(X_train, y_train)
print("Univariate Selection (Chi-squared):")
print(select_k_best.scores_)
print(X_train.columns)

# 2. Recursive Feature Elimination (RFE)
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_train, y_train)
print("\nRecursive Feature Elimination (RFE):")
print(rfe.support_)
print(rfe.ranking_)
print(X_train.columns)
# Create a DataFrame to display the results
rfe_results = pd.DataFrame({'Feature': X_train.columns, 'Selected': rfe.support_, 'Ranking': rfe.ranking_})

# Display the selected features and their rankings
selected_features = rfe_results[rfe_results['Selected'] == True]

print("Selected Features and Their Rankings:")
print(selected_features)

# Display all features sorted by their ranking
print("\nAll Features Sorted by Ranking:")
print(rfe_results.sort_values(by='Ranking'))

# 3. Feature Importance with RandomForest
model.fit(X_train, y_train)
print("\nFeature Importance (RandomForest):")
print(model.feature_importances_)
print(X_train.columns)

# 4. L1-based feature selection (Lasso)
lasso = LassoCV().fit(X_train, y_train)
importance = np.abs(lasso.coef_)
print("\nLasso Feature Importance:")
print(importance)
print(X_train.columns)

# Select features based on importance scores
selected_features = X_train.columns[(importance > np.mean(importance))]
print("\nSelected Features (Lasso):")
print(selected_features)

info_gain = mutual_info_classif(X, y)

# Create a DataFrame to display the information gain
info_gain_df = pd.DataFrame({'Feature': X.columns, 'Information Gain': info_gain})

# Sort the features by information gain
info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)

print(info_gain_df)

'''

'''----------------------------------------------'''

'''
# Train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy LogReg: {accuracy:.2f}')
print('Classification Report:')
print(report)

# MLP
model_n = MLPClassifier(hidden_layer_sizes=(128), max_iter=1000, random_state=42)
model_n.fit(X_train, y_train)
y_pred_n = model_n.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_n)
accuracy_t = accuracy_score(y_train, model_n.predict(X_train))
report = classification_report(y_test, y_pred_n)

print(f'Accuracy MLP: {accuracy:.2f}')
print(f'Accuracy Train: {accuracy_t:.2f}')
print('Classification Report:')
print(report)

# SVM

# Train an SVM model
model_s = SVC()
model_s.fit(X_train, y_train)

# Make predictions on the test set
y_pred_s = model_s.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_s)
report = classification_report(y_test, y_pred_s)

print(f'Accuracy SVM: {accuracy:.2f}')
print('Classification Report:')
print(report)

'''


#RF
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf)
report = classification_report(y_test, y_pred_rf)

print(f'Accuracy RF: {accuracy:.2f}')
print('Classification Report:')
print(report)

#XGBoost
# Train an XGBoost model
model_xg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xg = model_xg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_xg)
report = classification_report(y_test, y_pred_xg)

print(f'Accuracy XG: {accuracy:.2f}')
print('Classification Report:')
print(report)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy LogReg: {accuracy:.2f}')
print('Classification Report:')
print(report)

# MLP
model_n = MLPClassifier(hidden_layer_sizes=(128), max_iter=1000, random_state=42)
model_n.fit(X_train, y_train)
y_pred_n = model_n.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_n)
accuracy_t = accuracy_score(y_train, model_n.predict(X_train))
report = classification_report(y_test, y_pred_n)

print(f'Accuracy MLP: {accuracy:.2f}')
print(f'Accuracy Train: {accuracy_t:.2f}')
print('Classification Report:')
print(report)


# CV
k = 5  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#model = LogisticRegression()

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_rf, X, y, cv=kf, scoring='accuracy')

print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
print(f'Standard deviation of cross-validation score: {cv_scores.std():.2f}')
