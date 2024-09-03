import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

def read_csv_files(file_paths):
    df_dict = {}
    for file_path in file_paths:
        algo_name = os.path.basename(file_path).split('.out')[0]
        df_dict[algo_name] = pd.read_csv(file_path)
    return df_dict

'''
input_directory_dilithium_perf = "../../Downloads/MLClassification/liboqs_circl(dilithium)/dilithium_perf"
input_directory_dilithium_mem = "../../Downloads/MLClassification/liboqs_circl(dilithium)/dilithium_mem"
input_directory_dilithium_perf_circl = "../../Downloads/MLClassification/liboqs_circl(dilithium)/dilithium_perf_circl"
input_directory_dilithium_mem_circl = "../../Downloads/MLClassification/liboqs_circl(dilithium)/dilithium_mem_circl"
'''


input_directory_dilithium_perf = "../../Downloads/MLClassification/liboqs_circl(dilithium)/noisy/dilithium_perf"
input_directory_dilithium_mem = "../../Downloads/MLClassification/liboqs_circl(dilithium)/noisy/dilithium_mem"
input_directory_dilithium_perf_circl = "../../Downloads/MLClassification/liboqs_circl(dilithium)/noisy/dilithium_perf_circl"
input_directory_dilithium_mem_circl = "../../Downloads/MLClassification/liboqs_circl(dilithium)/noisy/dilithium_mem_circl"



dilithium_cycles = [os.path.join(input_directory_dilithium_perf, file) for file in os.listdir(input_directory_dilithium_perf) if file.endswith('.csv')]
dilithium_cycles_circl = [os.path.join(input_directory_dilithium_perf_circl, file) for file in os.listdir(input_directory_dilithium_perf_circl) if file.endswith('.csv')]
dilithium_mem = [os.path.join(input_directory_dilithium_mem, file) for file in os.listdir(input_directory_dilithium_mem) if file.endswith('.csv')]
dilithium_mem_circl = [os.path.join(input_directory_dilithium_mem_circl, file) for file in os.listdir(input_directory_dilithium_mem_circl) if file.endswith('.csv')]

dilithium_files_cycles = read_csv_files(dilithium_cycles)
dilithium_files_cycles_circl = read_csv_files(dilithium_cycles_circl)
dilithium_files_mem = read_csv_files(dilithium_mem)
dilithium_files_mem_circl = read_csv_files(dilithium_mem_circl)



merged_dilithium_dataframes = []
merged_dilithium_dataframes_circl = []

for algo_name, df1 in dilithium_files_cycles.items():
    if algo_name in dilithium_files_mem:
        df2 = dilithium_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_dilithium_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in dilithium_files_cycles_circl.items():
    if algo_name in dilithium_files_mem_circl:
        df2 = dilithium_files_mem_circl[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_dilithium_dataframes_circl.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")


dilithium_samples = pd.concat([df.sample(n=1000, random_state=42) for df in merged_dilithium_dataframes])

patterns = [
    (r'^[dD]ilithium.*', 'dilithium'),
]
for pattern, replacement in patterns:
    dilithium_samples['label'] = dilithium_samples['label'].astype(str).str.replace(pattern, replacement, regex=True)

dilithium_circl_samples = pd.concat([df.sample(n=1000, random_state=42) for df in merged_dilithium_dataframes_circl])

patterns = [
    (r'^[dD]ilithium.*', 'dilithium_circl'),
]
for pattern, replacement in patterns:
    dilithium_circl_samples['label'] = dilithium_circl_samples['label'].astype(str).str.replace(pattern, replacement, regex=True)

final_dataset = pd.concat([dilithium_samples, dilithium_circl_samples])
final_dataset = final_dataset.sample(frac=1)
final_dataset.fillna(0, inplace=True)


final_dataset = final_dataset.drop(['elapsed_time'], axis=1)
scaler = MinMaxScaler()
columns_to_scale = [col for col in final_dataset.columns if col != 'label']
final_dataset[columns_to_scale] = scaler.fit_transform(final_dataset[columns_to_scale])
final_dataset = final_dataset.sample(frac=1)

label_mapping = {label: idx for idx, label in enumerate(final_dataset['label'].unique())}
final_dataset['label'] = final_dataset['label'].map(label_mapping)

for label, idx in label_mapping.items():
    print(f"{label}: {idx}")
label_counts = final_dataset['label'].value_counts()
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

#X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'VmExe', 'CPU0_cycles']]
X = final_dataset[columns_to_scale]
y = final_dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

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