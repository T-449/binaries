import numpy as np
import pandas as pd
import os

from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def read_csv_files(file_paths):
    df_dict = {}
    for file_path in file_paths:
        if '_win' not in file_path:
            algo_name = os.path.basename(file_path).split('.out')[0]
        else:
            if '_win2' in file_path:
                algo_name = os.path.basename(file_path).split('.out')[0]
            elif 'perf' in file_path:
                algo_name = os.path.basename(file_path).split('_perf_output')[0]
            else:
                algo_name = os.path.basename(file_path).split('_mem_output')[0]
            algo_name = algo_name + '_win'
        df_dict[algo_name] = pd.read_csv(file_path)
    return df_dict

input_directory = ''
input_directory_mem = ''
#input_directory = ''
#input_directory_mem = ''
kyber_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('kyber')]
frodo_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('frodokem')]
bike_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('bike')]
#mceliece_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
#                    file.startswith('classic') and 'fast' not in file]
mceliece_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                       file.startswith('classic')]
ntru_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('ntruprime')]
hqc_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('hqc')]
sike_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('sike')]
rsa_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('rsa')]
ecdh_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                    file.startswith('ecdh')]

kyber_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('kyber')]
frodo_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('frodokem')]
bike_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('bike')]
#mceliece_file_paths = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
#                    file.startswith('classic') and 'fast' not in file]
mceliece_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                       file.startswith('classic')]
ntru_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('ntruprime')]
hqc_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('hqc')]
sike_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('sike')]


rsa_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('rsa')]
ecdh_file_paths_mem = [os.path.join(input_directory_mem, file) for file in os.listdir(input_directory_mem) if
                    file.startswith('ecdh')]


kyber_files_cycles = read_csv_files(kyber_file_paths)
mceliece_files_cycles = read_csv_files(mceliece_file_paths)
frodo_files_cycles = read_csv_files(frodo_file_paths)
bike_files_cycles = read_csv_files(bike_file_paths)
ntru_files_cycles = read_csv_files(ntru_file_paths)
hqc_files_cycles = read_csv_files(hqc_file_paths)
sike_files_cycles = read_csv_files(sike_file_paths)

rsa_files_cycles = read_csv_files(rsa_file_paths)
ecdh_files_cycles = read_csv_files(ecdh_file_paths)


kyber_files_mem = read_csv_files(kyber_file_paths_mem)
mceliece_files_mem = read_csv_files(mceliece_file_paths_mem)
frodo_files_mem = read_csv_files(frodo_file_paths_mem)
bike_files_mem = read_csv_files(bike_file_paths_mem)
ntru_files_mem = read_csv_files(ntru_file_paths_mem)
hqc_files_mem = read_csv_files(hqc_file_paths_mem)
sike_files_mem = read_csv_files(sike_file_paths_mem)

rsa_files_mem = read_csv_files(rsa_file_paths_mem)
ecdh_files_mem = read_csv_files(ecdh_file_paths_mem)

merged_kyber_dataframes = []
merged_mceliece_dataframes = []
merged_frodo_dataframes = []
merged_bike_dataframes = []
merged_ntru_dataframes = []
merged_hqc_dataframes = []
merged_sike_dataframes = []

merged_rsa_dataframes = []
merged_ecdh_dataframes = []

for algo_name, df1 in kyber_files_cycles.items():
    if algo_name in kyber_files_mem:
        df2 = kyber_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_kyber_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in mceliece_files_cycles.items():
    if algo_name in mceliece_files_mem:
        df2 = mceliece_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_mceliece_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in bike_files_cycles.items():
    if algo_name in bike_files_mem:
        df2 = bike_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_bike_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in frodo_files_cycles.items():
    if algo_name in frodo_files_mem:
        df2 = frodo_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_frodo_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in sike_files_cycles.items():
    if algo_name in sike_files_mem:
        df2 = sike_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_sike_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in hqc_files_cycles.items():
    if algo_name in hqc_files_mem:
        df2 = hqc_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_hqc_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in ntru_files_cycles.items():
    if algo_name in ntru_files_mem:
        df2 = ntru_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_ntru_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")


for algo_name, df1 in rsa_files_cycles.items():
    if algo_name in rsa_files_mem:
        df2 = rsa_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_rsa_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")


for algo_name, df1 in ecdh_files_cycles.items():
    if algo_name in ecdh_files_mem:
        df2 = ecdh_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_ecdh_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

file_lengths = {
    'kyber_files': len(merged_kyber_dataframes),
    'mceliece_files': len(merged_mceliece_dataframes),
    'frodo_files': len(merged_frodo_dataframes),
    'bike_files': len(merged_bike_dataframes),
    'ntru_files': len(merged_ntru_dataframes),
    'hqc_files': len(merged_hqc_dataframes),
    'sike_files': len(merged_sike_dataframes),
    'rsa_files': len(merged_rsa_dataframes),
    'ecdh_files': len(merged_ecdh_dataframes),
}

minimal = min(file_lengths.values())

kyber_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_kyber_dataframes), random_state=42) for df in merged_kyber_dataframes])
mceliece_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_mceliece_dataframes), random_state=42) for df in merged_mceliece_dataframes])
frodo_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_frodo_dataframes), random_state=42) for df in merged_frodo_dataframes])
bike_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_bike_dataframes), random_state=42) for df in merged_bike_dataframes])
ntru_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_ntru_dataframes), random_state=42) for df in merged_ntru_dataframes])
hqc_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_hqc_dataframes), random_state=42) for df in merged_hqc_dataframes])
sike_samples = pd.concat([df.sample(n=(1000 * minimal) // len(merged_sike_dataframes), random_state=42) for df in merged_sike_dataframes])

rsa_samples = pd.concat([df.sample(n=2000 // len(merged_rsa_dataframes), random_state=42) for df in merged_rsa_dataframes])
ecdh_samples = pd.concat([df.sample(n=2000 // len(merged_ecdh_dataframes), random_state=42) for df in merged_ecdh_dataframes])

#final_dataset = pd.concat([kyber_samples, mceliece_samples, frodo_samples, bike_samples, ntru_samples, hqc_samples, sike_samples, rsa_samples, ecdh_samples])
final_dataset = pd.concat([kyber_samples, mceliece_samples, frodo_samples, bike_samples, ntru_samples, hqc_samples, sike_samples])
final_dataset = final_dataset.sample(frac=1)
final_dataset.fillna(0, inplace=True)

patterns = [
    (r'^sikep.*', 'sike'),
    (r'^rsa.*', 'rsa'),
]
for pattern, replacement in patterns:
    final_dataset['label'] = final_dataset['label'].str.replace(pattern, replacement, regex=True)

output_file = './output.csv'
final_dataset.to_csv(output_file, index=False)

label_mapping = {label: idx for idx, label in enumerate(final_dataset['label'].unique())}
final_dataset['label'] = final_dataset['label'].map(label_mapping)
final_dataset = final_dataset.drop(['elapsed_time'], axis=1)
scaler = MinMaxScaler()
columns_to_scale = [col for col in final_dataset.columns if col != 'label']
final_dataset[columns_to_scale] = scaler.fit_transform(final_dataset[columns_to_scale])
final_dataset = final_dataset.sample(frac=1)

output_file = './output.csv'
final_dataset.to_csv(output_file, index=False)

for label, idx in label_mapping.items():
    print(f"{label}: {idx}")
label_counts = final_dataset['label'].value_counts()
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

#X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'VmExe', 'VmLib', 'VmPTE', 'CPU5_cycles', 'CPU11_cycles']]
#X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'VmExe', 'VmLib']]
X = final_dataset[['VmSize', 'VmData', 'VmRSS', 'VmExe']]
#X = final_dataset[columns_to_scale]
y = final_dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




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










#LogReg
model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy Log Reg: {accuracy:.2f}')
print('Classification Report:')
print(report)



#XGBoost
model_xg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model_xg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xg = model_xg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_xg)
report = classification_report(y_test, y_pred_xg, target_names=label_mapping.keys())

print(f'XGBoost Accuracy: {accuracy:.2f}')
print('XGBoost Classification Report:')
print(report)

#MLP

mlp_model = MLPClassifier(hidden_layer_sizes=(128, 128), random_state=42, max_iter=1000)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
report_mlp = classification_report(y_test, y_pred_mlp, target_names=[str(key) for key in label_mapping.keys()])

print(f'MLP Classifier Accuracy: {accuracy_mlp:.2f}')
print('MLP Classifier Report:')
print(report_mlp)

#RF

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=[str(key) for key in label_mapping.keys()])

print(f'Random Forest Classifier Accuracy: {accuracy_rf:.2f}')
print('Random Forest Classifier Report:')
print(report_rf)

#SVM
# Train and evaluate an SVM model for multiclass classification
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm, target_names=[str(key) for key in label_mapping.keys()])

print(f'SVM Classifier Accuracy: {accuracy_svm:.2f}')
print('SVM Classifier Report:')
print(report_svm)


output_file = 'output.csv'
final_dataset.to_csv(output_file, index=False)
