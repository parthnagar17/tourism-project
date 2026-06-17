import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/parth1706/tourism-package-predictions/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#Data Cleaning
df.drop_duplicates(inplace=True)
columns_to_drop = ["CustomerID"]
df.drop(columns=columns_to_drop, inplace=True)

#Handle Missing Values
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

target_col = 'ProdTaken'
# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="parth1706/tourism-package-predictions",
        repo_type="dataset",
    )

print("Files uploaded successfully!")
