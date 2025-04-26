import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances

# Load the original and protected datasets
original = pd.read_csv("original.csv")
protected = pd.read_csv("protected.csv")

# Define features to be used (exclude Identifier and Name)
features = original.columns.drop(["Identifier", "Name"])
numeric_features = original[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = original[features].select_dtypes(include=["object"]).columns.tolist()

# Set up preprocessing pipeline: scale numeric and one-hot encode categorical
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Apply preprocessing
original_processed = pipeline.fit_transform(original[features])
protected_processed = pipeline.transform(protected[features])

# Matching all 20,000 rows using batch-wise distance computation
batch_size = 500
matches = []

print("Matching rows in batches...")
for start in range(0, protected_processed.shape[0], batch_size):
    end = min(start + batch_size, protected_processed.shape[0])
    batch = protected_processed[start:end]
    distances = euclidean_distances(batch, original_processed)
    closest = np.argmin(distances, axis=1)
    matches.extend(closest)
    print(f"Processed rows {start} to {end}")

# Save match results to CSV (optional)
matched_df = pd.DataFrame({
    "Protected_Name": protected["Name"],
    "Predicted_Original_Name": original.loc[matches, "Name"].values,
    "Protected_Identifier": protected["Identifier"],
    "Predicted_Original_Identifier": original.loc[matches, "Identifier"].values
})
matched_df.to_csv("matched_results.csv", index=False)
print("\n✅ Matching complete. Results saved to 'matched_results.csv'.")

# Preview top 10 matches
print("\nSample Matches (Protected → Predicted Original):")
for i in range(10):
    print(f"{matched_df.iloc[i]['Protected_Name']} → {matched_df.iloc[i]['Predicted_Original_Name']}")
