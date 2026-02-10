import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from config.paths import PREPROCESSED_TRAIN_PATH, MODEL_PATH, PREPROCESSOR_PATH


def train_data(df):
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    feature_columns = X_train.columns.to_list()

    print("Training the model....")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    print("training completed...")

    # Bundle model, features, and preprocessor into single artifact
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    artifact = {
        "model": model,
        "features": feature_columns,
        "preprocessor": preprocessor
    }
    joblib.dump(artifact, MODEL_PATH)

    return True



if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    train_data(df_train)