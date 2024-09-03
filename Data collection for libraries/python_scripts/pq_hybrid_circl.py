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


def merge_cycle_mem(file_list_1, file_list_2, hybrid = 0):
    merged_algo_df = []
    for file1 in file_list_1:
        algo_name1 = os.path.basename(file1).split('.out')[0]
        for file2 in file_list_2:
            algo_name2 = os.path.basename(file2).split('.out')[0]
            if algo_name1 == algo_name2:
                # print(file1, file2)
                if hybrid == 0:
                    df1 = pd.read_csv(input_dir_cycles + '/' + file1)
                    df2 = pd.read_csv(input_dir_mem + '/' + file2)
                else:
                    df1 = pd.read_csv(input_dir_cycles_hybrid + '/' + file1)
                    df2 = pd.read_csv(input_dir_mem_hybrid + '/' + file2)
                merged_df = pd.concat([df1, df2], axis=1)

                merged_algo_df.append(merged_df)
    return merged_algo_df


input_dir_cycles = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/clean/pq_perf'
input_dir_mem = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/clean/pq_mem'
input_dir_cycles_hybrid = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/clean/hybrid_perf'
input_dir_mem_hybrid = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/clean/hybrid_mem'




'''
input_dir_cycles = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/noisy/pq_perf'
input_dir_mem = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/noisy/pq_mem'
input_dir_cycles_hybrid = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/noisy/hybrid_perf'
input_dir_mem_hybrid = '/home/heresy/Downloads/MLClassification/pq vs hybrid_circl/noisy/hybrid_mem'
'''




frodo_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                              file.startswith('frodo')]
frodo_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                           file.startswith('frodo')]
merged_frodo_df = merge_cycle_mem(frodo_file_path_cycles, frodo_file_path_mem)


kyber_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                              file.startswith('kyber')]
kyber_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                           file.startswith('kyber')]
merged_kyber_df = merge_cycle_mem(kyber_file_path_cycles, kyber_file_path_mem)


sike_file_path_cycles = [file for file in os.listdir(input_dir_cycles) if
                              file.startswith('sike')]
sike_file_path_mem = [file for file in os.listdir(input_dir_mem) if
                           file.startswith('sike')]
merged_sike_df = merge_cycle_mem(sike_file_path_cycles, sike_file_path_mem)


kyber512x25519_file_path_cycles = [file for file in os.listdir(input_dir_cycles_hybrid) if
                              file.startswith('kyber512x25519')]
kyber512x25519_file_path_mem = [file for file in os.listdir(input_dir_mem_hybrid) if
                           file.startswith('kyber512x25519')]
merged_kyber512x25519_df = merge_cycle_mem(kyber512x25519_file_path_cycles, kyber512x25519_file_path_mem,1 )



kyber768p256_file_path_cycles = [file for file in os.listdir(input_dir_cycles_hybrid) if
                              file.startswith('kyber768p256')]
kyber768p256_file_path_mem = [file for file in os.listdir(input_dir_mem_hybrid) if
                           file.startswith('kyber768p256')]
merged_kyber768p256_df = merge_cycle_mem(kyber768p256_file_path_cycles, kyber768p256_file_path_mem, 1)


kyber768x448_file_path_cycles = [file for file in os.listdir(input_dir_cycles_hybrid) if
                              file.startswith('kyber768x448')]
kyber768x448_file_path_mem = [file for file in os.listdir(input_dir_mem_hybrid) if
                           file.startswith('kyber768x448')]
merged_kyber768x448_df = merge_cycle_mem(kyber768x448_file_path_cycles, kyber768x448_file_path_mem, 1)


kyber768x25519_file_path_cycles = [file for file in os.listdir(input_dir_cycles_hybrid) if
                              file.startswith('kyber768x25519')]
kyber768x25519_file_path_mem = [file for file in os.listdir(input_dir_mem_hybrid) if
                           file.startswith('kyber768x25519')]
merged_kyber768x25519_df = merge_cycle_mem(kyber768x25519_file_path_cycles, kyber768x25519_file_path_mem, 1)


kyber1024x448_file_path_cycles = [file for file in os.listdir(input_dir_cycles_hybrid) if
                              file.startswith('kyber1024x448')]
kyber1024x448_file_path_mem = [file for file in os.listdir(input_dir_mem_hybrid) if
                           file.startswith('kyber1024x448')]
merged_kyber1024x448_df = merge_cycle_mem(kyber1024x448_file_path_cycles, kyber1024x448_file_path_mem, 1)


kyber_samples = pd.concat([df.sample(n=6000 // len(merged_kyber_df)) for df in merged_kyber_df])
frodo_samples = pd.concat([df.sample(n=3000 // len(merged_frodo_df)) for df in merged_frodo_df])
sike_samples = pd.concat([df.sample(n=6000 // len(merged_sike_df)) for df in merged_sike_df])
kyber768x25519_samples = pd.concat([df.sample(n=3000 // len(merged_kyber768x25519_df)) for df in merged_kyber768x25519_df])
kyber768x448_samples = pd.concat([df.sample(n=3000 // len(merged_kyber768x448_df)) for df in merged_kyber768x448_df])
kyber768p256_samples = pd.concat([df.sample(n=3000 // len(merged_kyber768p256_df)) for df in merged_kyber768p256_df])
kyber512x25519_samples = pd.concat([df.sample(n=3000 // len(merged_kyber512x25519_df)) for df in merged_kyber512x25519_df])
kyber1024x448_samples = pd.concat([df.sample(n=3000 // len(merged_kyber1024x448_df)) for df in merged_kyber1024x448_df])



pq_samples = pd.concat([kyber_samples, frodo_samples, sike_samples])
hybrid_samples = pd.concat([kyber512x25519_samples, kyber768x448_samples, kyber768x25519_samples, kyber768p256_samples, kyber1024x448_samples])
final_dataset = pd.concat([pq_samples, hybrid_samples])

final_dataset = final_dataset.drop(['elapsed_time'], axis=1)
final_dataset.fillna(0, inplace=True)

patterns = [
    (r'^frodo.*', 'pq'),
    (r'^kyber$', 'pq'),
    (r'^sikep.*', 'pq'),
    (r'^kyber.+', 'hybrid')
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


#X = final_dataset[columns_to_scale]
X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'VmExe']]
# X = final_dataset[['max_value']]
y = final_dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy Logistic Regression: {accuracy:.2f}')
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


#RF
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf)
report = classification_report(y_test, y_pred_rf)

print(f'Accuracy Random Forest: {accuracy:.2f}')
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

print(f'Accuracy XGBoost: {accuracy:.2f}')
print('Classification Report:')
print(report)
