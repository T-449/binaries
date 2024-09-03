import numpy as np
import pandas as pd
import os

from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

'''
def merge_cycle_mem(file_list_1, file_list_2):
    merged_algo_df = []
    win = 0
    for file1 in file_list_1:
        if '_win' in file1:
            algo_name1 = os.path.basename(file1).split('_perf_output')[0]
            for file2 in file_list_2:
                if '_win' in file2:
                    algo_name2 = os.path.basename(file2).split('_mem_output')[0]
        else:
            algo_name1 = os.path.basename(file1).split('.out')[0]
            for file2 in file_list_2:
                if '_win' not in  file2:
                    algo_name2 = os.path.basename(file2).split('.out')[0]
        if algo_name1 == algo_name2:
            df1 = pd.read_csv(input_directory + '/' + file1)
            df2 = pd.read_csv(input_directory_mem + '/' + file2)
            merged_df = pd.concat([df1, df2], axis=1)

            merged_algo_df.append(merged_df)

    return merged_algo_df
'''




def merge_cycle_mem(file_list_1, file_list_2):
    merged_algo_df = []
    for file1 in file_list_1:
        algo_name1 = os.path.basename(file1).split('.out')[0]
        for file2 in file_list_2:
            algo_name2 = os.path.basename(file2).split('.out')[0]
            if algo_name1 == algo_name2:
                # print(file1, file2)
                df1 = pd.read_csv(input_directory + '/' + file1)
                df2 = pd.read_csv(input_directory_mem + '/' + file2)
                merged_df = pd.concat([df1, df2], axis=1)

                merged_algo_df.append(merged_df)
    return merged_algo_df

input_directory = '../../Downloads/MLClassification/csv/KEM_perf'
#input_directory = '../../Downloads/CSV/KEM_perf'
input_directory_mem = '../../Downloads/MLClassification/csv/KEM_mem'
#input_directory_mem = '../../Downloads/CSV/KEM_mem'


kyber_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('kyber')]
kyber_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('kyber')]
merged_kyber_df = merge_cycle_mem(kyber_file_paths_cycles, kyber_file_paths_mem)


bike_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('bike')]
bike_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('bike')]
merged_bike_df = merge_cycle_mem(bike_file_paths_cycles, bike_file_paths_mem)


sike_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('sike')]
sike_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('sike')]
merged_sike_df = merge_cycle_mem(sike_file_paths_cycles, sike_file_paths_mem)


mceliece_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('classic')]
mceliece_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('classic')]
merged_mceliece_df = merge_cycle_mem(mceliece_file_paths_cycles, mceliece_file_paths_mem)


frodo_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('frodo')]
frodo_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('frodo')]
merged_frodo_df = merge_cycle_mem(frodo_file_paths_cycles, frodo_file_paths_mem)


hqc_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('hqc')]
hqc_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('hqc')]
merged_hqc_df = merge_cycle_mem(hqc_file_paths_cycles, hqc_file_paths_mem)


ntru_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('ntru')]
ntru_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('ntru')]
merged_ntru_df = merge_cycle_mem(ntru_file_paths_cycles, ntru_file_paths_mem)


rsa_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('rsa')]
rsa_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('rsa')]
merged_rsa_df = merge_cycle_mem(rsa_file_paths_cycles, rsa_file_paths_mem)


ecdh_file_paths_cycles = [file for file in os.listdir(input_directory) if
                           file.startswith('ecdh')]
ecdh_file_paths_mem = [file for file in os.listdir(input_directory_mem) if
                        file.startswith('ecdh')]
merged_ecdh_df = merge_cycle_mem(ecdh_file_paths_cycles, ecdh_file_paths_mem)


file_lengths = {
    'kyber_files': len(merged_kyber_df),
    'mceliece_files': len(merged_mceliece_df),
    'frodo_files': len(merged_frodo_df),
    'bike_files': len(merged_bike_df),
    'ntru_files': len(merged_ntru_df),
    'hqc_files': len(merged_hqc_df),
    'sike_files': len(merged_sike_df),
}

minimal = min(file_lengths.values())

print(minimal)

pq_samples = 1000 * (len(merged_rsa_df) + len(merged_ecdh_df)) // 7

if pq_samples > 1000 * minimal:
    pq_samples = 1000 * minimal

print(pq_samples)

kyber_samples = pd.concat([df.sample(n=(pq_samples + 150) // len(merged_kyber_df), random_state=42) for df in merged_kyber_df])
mceliece_samples = pd.concat([df.sample(n=(pq_samples + 150) // len(merged_mceliece_df), random_state=42) for df in merged_mceliece_df])
frodo_samples = pd.concat([df.sample(n=(pq_samples + 150) // len(merged_frodo_df), random_state=42) for df in merged_frodo_df])
bike_samples = pd.concat([df.sample(n=(pq_samples + 150) // len(merged_bike_df), random_state=42) for df in merged_bike_df])
ntru_samples = pd.concat([df.sample(n=pq_samples // len(merged_ntru_df), random_state=42) for df in merged_ntru_df])
hqc_samples = pd.concat([df.sample(n=(pq_samples + 150) // len(merged_hqc_df), random_state=42) for df in merged_hqc_df])
sike_samples = pd.concat([df.sample(n=(pq_samples + 150) // len(merged_sike_df), random_state=42) for df in merged_sike_df])
rsa_samples = pd.concat(df for df in merged_rsa_df)
ecdh_samples = pd.concat(df for df in merged_ecdh_df)

# Combine the samples from classical files and quantum files
final_dataset = pd.concat([kyber_samples, mceliece_samples, frodo_samples, bike_samples, ntru_samples, hqc_samples, sike_samples, rsa_samples, ecdh_samples])
final_dataset = final_dataset.drop(['elapsed_time'], axis=1)
final_dataset.fillna(0, inplace=True)

patterns = [
    (r'^sikep.*', 'sike'),
    (r'^rsa.*', 'rsa'),
    (r'^frodo.*', 'frodo'),
]
for pattern, replacement in patterns:
    final_dataset['label'] = final_dataset['label'].str.replace(pattern, replacement, regex=True)


scaler = MinMaxScaler()
label_column = 'label'  # Replace with the name of your label column if different
columns_to_scale = [col for col in final_dataset.columns if col != label_column]


# Apply Min-Max scaling to all columns except the label

label_counts = final_dataset['label'].value_counts()
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")


final_dataset.fillna(0, inplace=True)



final_dataset[columns_to_scale] = scaler.fit_transform(final_dataset[columns_to_scale])
final_dataset['label'] = final_dataset['label'].apply(lambda x: 0 if x in ['rsa', 'ecdh'] else 1)
final_dataset = final_dataset.sample(frac=1)

label_counts = final_dataset['label'].value_counts()
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

print(final_dataset['label'].value_counts())

output_file = './output.csv'
final_dataset.to_csv(output_file, index=False)

#print(final_dataset.label.value_counts())

#X = final_dataset[columns_to_scale]
X = final_dataset[['VmSize', 'VmData', 'VmRSS', 'VmExe']]
# X = final_dataset[['max_value']]
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
'''




'''----------------------------------------------'''








# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy LogReg: {accuracy:.2f}')
print('Classification Report:')
print(report)

# MLP
model_n = MLPClassifier(hidden_layer_sizes=(128), max_iter=100, random_state=42)
model_n.fit(X_train, y_train)
y_pred_n = model_n.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_n)
accuracy_t = accuracy_score(y_train, model_n.predict(X_train))
report = classification_report(y_test, y_pred_n)

print(f'Accuracy MLP: {accuracy:.2f}')
print(f'Accuracy Train: {accuracy_t:.2f}')
print('Classification Report:')
print(report)

'''
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

# CV
k = 5  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#model = LogisticRegression()

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_rf, X, y, cv=kf, scoring='accuracy')

print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
print(f'Standard deviation of cross-validation score: {cv_scores.std():.2f}')

# Save the final dataset to a new CSV file


#print(f"Final dataset saved to {output_file}")
