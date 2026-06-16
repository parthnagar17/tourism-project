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


#Split into Train/Test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

#Save Locally
os.makedirs("tourism_project_lcl/data/processed", exist_ok=True)

train_df.to_csv(
    "tourism_project_lcl/data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "tourism_project_lcl/data/processed/test.csv",
    index=False
)

print("Files saved successfully.")

# Upload Train/Test Back to Hugging Face
repo_id = "parth1706/tourism-package-predictions"
repo_type = "dataset"

api.upload_folder(
    folder_path="tourism_project_lcl/data/processed/",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Files uploaded successfully!")
