import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from config.paths import PREPROCESSED_TRAIN_PATH, MODEL_PATH, FEATURE_STORE_PATH


def train_data(df):
    # df = df.dropna()
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    # save features before training
    feature_columns = X_train.columns.to_list()
    joblib.dump(feature_columns, FEATURE_STORE_PATH)

    print("Training the model....")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    print("training completed...")
    joblib.dump(model, MODEL_PATH)

    return True



if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    train_data(df_train)