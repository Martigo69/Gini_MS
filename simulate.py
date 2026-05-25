

import pandas as pd
import numpy as np

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
# MS-GINI Score Computation
# =====================================================================

def compute_ms_gini_score(dataset, target_col, scale_factor=2**16):
    """
    Compute MS-GINI scores - SCALED VERSION matching MPC
    """
    m = dataset.shape[0]
    label_col = target_col
    classes = dataset[label_col].unique()
    n = len(classes)

    print(classes.shape)
    
    L = pd.get_dummies(dataset[label_col], drop_first=True).to_numpy()

    print(L.shape)

    features_df = dataset.drop(columns=[target_col])
    scaled_features = features_df * scale_factor

    print(scaled_features.shape)
    
    gini_scores = {}
    
    for feature in features_df.columns:
        theta = scaled_features[feature].mean()
        
        a = 0
        b = 0
        A = {cls: 0 for cls in range(n-1)}
        B = {cls: 0 for cls in range(n-1)}

        for i in range(m):
            flags = theta < scaled_features[feature].iloc[i]
            b += flags
            for j in range(n-1):
                flagm = flags * L[i][j]
                B[j] = B[j] + flagm
                A[j] = A[j] + L[i][j] - flagm

        a = m - b
        A[n-1] = a - sum(A[cls] for cls in range(n-1))
        B[n-1] = b - sum(B[cls] for cls in range(n-1))

        sum_A_sq = sum(A[cls] ** 2 for cls in range(n))
        sum_B_sq = sum(B[cls] ** 2 for cls in range(n))

        G_le = a - (sum_A_sq / a) if a > 0 else 0
        G_gt = b - (sum_B_sq / b) if b > 0 else 0
        
        G_F = G_le + G_gt
        
        gini_scores[feature] = {
            'score': G_F,
            'theta': theta /  2**16,  
            'a': a,
            'b': b,
            'A': A,
            'B': B,
            'G_le': G_le,
            'G_gt': G_gt,
        }
    
    return gini_scores


# =====================================================================
# Main Execution
# =====================================================================


if __name__ == "__main__":
    dataset_name = 'wdbc'  
    target_col = 'Diagnosis'

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

    gini_scores = compute_ms_gini_score(dataset_df, target_col, scale_factor=2**16)
    
    print(f"MS-GINI Scores for {dataset_name} dataset (sorted by score - lower is better):")
    print("-" * 100)
    print(f"{'Feature':<20} {'Score':>15} {'Theta (Mean)':>15} {'a':>8} {'b':>8} {'G(S<=θ)':>15} {'G(S>θ)':>15}")
    print("-" * 100)
    # Sort by the order of columns in the dataset
    feature_order = list(dataset_df.drop(columns=[target_col]).columns)
    for feature, info in sorted(gini_scores.items(), key=lambda x: feature_order.index(x[0])):
        print(
            f"{feature:<20} {info['score']:>20.17g} {info['theta']:>20.17g} "
            f"{info['a']:>8} {info['b']:>8} {info['G_le']:>20.17g} {info['G_gt']:>20.17g} "
        )

    print("-" * 100)
    # sort features by score
    sorted_features = sorted(gini_scores.items(), key=lambda x: x[1]['score'])
    print("Features sorted by MS-GINI score:")
    for feature, info in sorted_features:
        print(f"{feature}: Score = {info['score']:.17g}")