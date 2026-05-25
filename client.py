import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# =====================================================================
# Dataset Loading Functions 
# =====================================================================

def load_beans_data():
    df = pd.read_csv('beans_kmeans.csv', delimiter=',')
    df = df.apply(lambda x: x.astype(np.int64) if x.name != 'Class' else x)
    return df, 'Class'

def load_diabetes_data():
    df = pd.read_csv('diabetes_kmeans.csv', delimiter=',')
    df = df.apply(lambda x: x.astype(np.int64) if x.name != 'Outcome' else x)
    return df, 'Outcome'

def load_divorce_data():
    df = pd.read_csv('divorce.csv', delimiter=',')
    df = df.apply(lambda x: x.astype(np.int64) if x.name != 'Class' else x)
    return df, 'Class'

def load_parkinsons_data():
    df = pd.read_csv('parkinsons_kmeans.csv', delimiter=',')
    df = df.drop(columns=['name'], errors='ignore')
    df = df.apply(lambda x: x.astype(np.int64) if x.name != 'status' else x)
    return df, 'status'

def load_rice_binned_data():
    df = pd.read_csv('rice_binned_kmeans.csv', delimiter=',')
    df = df.apply(lambda x: x.astype(np.int64) if x.name != 'Class' else x)
    return df, 'Class'

def load_wdbc_binned_data():
    df = pd.read_csv('wdbc_binned_kmeans.csv', delimiter=',')
    df = df.drop(columns=['ID'], errors='ignore')
    df = df.apply(lambda x: x.astype(np.int64) if x.name != 'Diagnosis' else x)
    return df, 'Diagnosis'

# =====================================================================
# Main Processing Function
# =====================================================================

def process_dataset(dataset_df, target_col, dataset_name):
    SCALE = 2**16

    X = dataset_df.drop(columns=[target_col]).to_numpy().astype(float)
    y = dataset_df[target_col].to_numpy()

    L = pd.get_dummies(y).to_numpy()

    if L.shape[1] == 2:
        L = L[:, :1]
    else:
        L = L[:, :-1]

    X_scaled = (X * SCALE).astype(np.uint64)
    L_unscaled = L.astype(np.uint64)

    os.makedirs("Player-Data", exist_ok=True)
    
    values = np.concatenate([
        X_scaled.flatten(),
        L_unscaled.flatten()  
    ])
    
    path = "Player-Data/Input-P0-0-" + dataset_name
    with open(path, "w") as fh:
        fh.write(" ".join(map(str, values)))
    
    print(f"Wrote {values.size} values to {path}")
    print(f"Dataset shape: {dataset_df.shape}, Target: {target_col}")
    print(f"Number of classes: {L.shape[1] + 1}")

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    dataset_name = 'diabetes'  # Change this to load different datasets
    
    if dataset_name == 'beans':
        dataset_df, target_col = load_beans_data()
    elif dataset_name == 'diabetes':
        dataset_df, target_col = load_diabetes_data()
    elif dataset_name == 'divorce':
        dataset_df, target_col = load_divorce_data()
    elif dataset_name == 'parkinsons':
        dataset_df, target_col = load_parkinsons_data()
    elif dataset_name == 'rice':
        dataset_df, target_col = load_rice_binned_data()
    elif dataset_name == 'wdbc':
        dataset_df, target_col = load_wdbc_binned_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    process_dataset(dataset_df, target_col, dataset_name)


