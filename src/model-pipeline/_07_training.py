import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"

ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_v1"
os.makedirs(ARTIFACT_PATH, exist_ok=True)

MODEL_ARTIFACT = ARTIFACT_PATH / "model.pkl"
FEATURE_STORE = ARTIFACT_PATH / "features.pkl"


def trianing_data(df):
    # df = df.dropna()
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    # save features before training
    feature_columns = X_train.columns.to_list()
    joblib.dump(feature_columns, FEATURE_STORE)

    print("Training the model....")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    print("training completed...")
    joblib.dump(model, MODEL_ARTIFACT)

    return True



if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    trianing_data(df_train)