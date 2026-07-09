import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def find_neighbors(csv_path, query, k, feature_cols = None, id_col = "Participant_ID"):
    
    df = pd.read_csv(csv_path)

    feature_cols = list(query.keys())

    # wenn keine Feature‑Liste übergeben: automatisch numerische Spalten (außer ID)
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != id_col]

    needed = feature_cols + [id_col]
    df = df[needed].dropna().reset_index(drop=True)

    X = df[feature_cols].values               # (n_samples, n_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(X_scaled)

    print("NN trained.")

    query_vec = np.array([[query[col] for col in feature_cols]])   # 1 × n_features
    query_vec_scaled = scaler.transform(query_vec)

    distances, indices = nn.kneighbors(query_vec_scaled, n_neighbors=k, return_distance=True)

    neighbors_df = df.iloc[indices[0]].copy()

    return neighbors_df