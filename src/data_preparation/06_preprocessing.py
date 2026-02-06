import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.paths import FEATURED_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, PREPROCESSOR_PATH


def preprocess_data(df):
    df_pp = df.copy()

    # Separate features and target
    X = df_pp.drop(columns=['Attrition'])
    y = df_pp['Attrition']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize numerical features
    numeric_cols = ['Years at Company', 'Company Tenure', 'RoleStagnationRatio', 'TenureGap']

    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    joblib.dump(scaler, PREPROCESSOR_PATH)

    # 🔑 RESET INDICES (CRITICAL)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save preprocessed data
    train_df.to_csv(PREPROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PREPROCESSED_TEST_PATH, index=False)

    print(train_df.head())
    print("Preprocessing completed and data saved.")
    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv(FEATURED_PATH)
    preprocess_data(df)