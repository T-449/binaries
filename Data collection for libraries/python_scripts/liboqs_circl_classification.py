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


input_directory_kyber_perf = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Kyber_perf"
input_directory_kyber_mem = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Kyber_mem"
input_directory_frodo_perf = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/frodo_perf"
input_directory_frodo_mem = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/frodo_mem"

input_directory_kyber_perf_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Kyber_perf_circl"
input_directory_kyber_mem_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Kyber_mem_circl"
input_directory_frodo_perf_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/frodo_perf_circl"
input_directory_frodo_mem_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/frodo_mem_circl"



'''
input_directory_kyber_perf = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/Kyber_perf"
input_directory_kyber_mem = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/Kyber_mem"
input_directory_frodo_perf = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/frodo_perf"
input_directory_frodo_mem = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/frodo_mem"

input_directory_kyber_perf_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/Kyber_perf_circl"
input_directory_kyber_mem_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/Kyber_mem_circl"
input_directory_frodo_perf_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/frodo_perf_circl"
input_directory_frodo_mem_circl = "../../Downloads/MLClassification/liboqs_circl(frodo+kyber)/Tainted/frodo_mem_circl"
'''





kyber_cycles = [os.path.join(input_directory_kyber_perf, file) for file in os.listdir(input_directory_kyber_perf) if file.endswith('.csv')]
kyber_cycles_circl = [os.path.join(input_directory_kyber_perf_circl, file) for file in os.listdir(input_directory_kyber_perf_circl) if file.endswith('.csv')]
frodo_cycles = [os.path.join(input_directory_frodo_perf, file) for file in os.listdir(input_directory_frodo_perf) if file.endswith('.csv')]
frodo_cycles_circl = [os.path.join(input_directory_frodo_perf_circl, file) for file in os.listdir(input_directory_frodo_perf_circl) if file.endswith('.csv')]

kyber_mem = [os.path.join(input_directory_kyber_mem, file) for file in os.listdir(input_directory_kyber_mem) if file.endswith('.csv')]
kyber_mem_circl = [os.path.join(input_directory_kyber_mem_circl, file) for file in os.listdir(input_directory_kyber_mem_circl) if file.endswith('.csv')]
frodo_mem = [os.path.join(input_directory_frodo_mem, file) for file in os.listdir(input_directory_frodo_mem) if file.endswith('.csv')]
frodo_mem_circl = [os.path.join(input_directory_frodo_mem_circl, file) for file in os.listdir(input_directory_frodo_mem_circl) if file.endswith('.csv')]

kyber_files_cycles = read_csv_files(kyber_cycles)
kyber_files_cycles_circl = read_csv_files(kyber_cycles_circl)
frodo_files_cycles = read_csv_files(frodo_cycles)
frodo_files_cycles_circl = read_csv_files(frodo_cycles_circl)


kyber_files_mem = read_csv_files(kyber_mem)
kyber_files_mem_circl = read_csv_files(kyber_mem_circl)
frodo_files_mem = read_csv_files(frodo_mem)
frodo_files_mem_circl = read_csv_files(frodo_mem_circl)

merged_kyber_dataframes = []
merged_kyber_dataframes_circl = []
merged_frodo_dataframes = []
merged_frodo_dataframes_circl = []


for algo_name, df1 in kyber_files_cycles.items():
    if algo_name in kyber_files_mem:
        df2 = kyber_files_mem[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_kyber_dataframes.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

for algo_name, df1 in kyber_files_cycles_circl.items():
    if algo_name in kyber_files_mem_circl:
        df2 = kyber_files_mem_circl[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_kyber_dataframes_circl.append(merged_df)
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

print(len(merged_frodo_dataframes))

for algo_name, df1 in frodo_files_cycles_circl.items():
    if algo_name in frodo_files_mem_circl:
        df2 = frodo_files_mem_circl[algo_name]

        # Concatenate the DataFrames column-wise
        merged_df = pd.concat([df1, df2], axis=1)
        merged_frodo_dataframes_circl.append(merged_df)
    else:
        print(f"Warning: {algo_name} is missing in the memory files.")

kyber_samples = pd.concat([df.sample(n=3000  // len(merged_kyber_dataframes), random_state=42) for df in merged_kyber_dataframes])

kyber_circl_samples = pd.concat([df.sample(n=3000  // len(merged_kyber_dataframes_circl), random_state=42) for df in merged_kyber_dataframes_circl])

frodo_samples = pd.concat([df.sample(n=3000  // len(merged_frodo_dataframes), random_state=42) for df in merged_frodo_dataframes])

frodo_samples_circl = pd.concat([df.sample(n=3000  // len(merged_frodo_dataframes_circl), random_state=42) for df in merged_frodo_dataframes_circl])


final_dataset = pd.concat([kyber_samples, kyber_circl_samples, frodo_samples, frodo_samples_circl])
final_dataset = final_dataset.sample(frac=1)
final_dataset.fillna(0, inplace=True)


label_mapping = {label: idx for idx, label in enumerate(final_dataset['label'].unique())}
final_dataset['label'] = final_dataset['label'].map(label_mapping)
final_dataset = final_dataset.drop(['elapsed_time'], axis=1)
scaler = MinMaxScaler()
columns_to_scale = [col for col in final_dataset.columns if col != 'label']
final_dataset[columns_to_scale] = scaler.fit_transform(final_dataset[columns_to_scale])
final_dataset = final_dataset.sample(frac=1)

label_counts = final_dataset['label'].value_counts()
for label, idx in label_mapping.items():
    print(f"{label}: {idx}")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

#X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'VmExe', 'VmLib', 'VmPTE', 'CPU5_cycles', 'CPU11_cycles']]
#X = final_dataset[['VmSize', 'VmRSS', 'VmData', 'VmExe', 'VmLib']]
X = final_dataset[['VmSize', 'VmData', 'VmRSS', 'VmExe']]
#X = final_dataset[columns_to_scale]
y = final_dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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