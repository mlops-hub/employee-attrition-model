import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config.paths import PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, MODEL_PATH, METRICS_PATH


def tuning_data(X_train, y_train, X_test, y_test):

    # load base model
    model = joblib.load(MODEL_PATH)

    # set parameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'l1_ratio': [0],     # equivalent to L2
        'max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(model, param_grid=param_grid, cv=strat_cv, scoring='recall')

    # train gridsearchcv model
    grid.fit(X_train, y_train)

    # get best model and save in models/
    tuned_model = grid.best_estimator_

    #  replace previous model with best one.
    joblib.dump(tuned_model, MODEL_PATH)
    
    print(f'best params: {grid.best_params_}')
    print(f'best cv scores: {grid.best_score_ * 100}')
    print(f'best model: {tuned_model}')

    # log other paramaeter values in logs/
    results = pd.DataFrame(grid.cv_results_)
    print(results.head(3))
        
    # predict the output with tuned_model
    y_pred_ht = tuned_model.predict(X_test)

    # tuned model evaluation
    accuracy_ht = accuracy_score(y_test, y_pred_ht)
    recall_ht = recall_score(y_test, y_pred_ht)
    print('accuracy: ', accuracy_ht)
    print('recall: ', recall_ht)

    # get trin/test score
    tuned_train_score = tuned_model.score(X_train, y_train) 
    tuned_test_score =  tuned_model.score(X_test, y_test)

    print('tuned train score: ', tuned_train_score)
    print('tuned test score: ', tuned_test_score)

    # Compare CV scores of base model and tuned model
    base_cv_scores = cross_val_score(model, X_train, y_train, cv=strat_cv)
    tuned_cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=strat_cv)

    print("=== MODEL COMPARISON ===")
    print(f"Base Model CV:     {base_cv_scores.mean():.4f} (+/- {base_cv_scores.std():.4f})")
    print(f"Tuned Model CV:    {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std():.4f})")

    metrics = {
        "accuracy": accuracy_ht,
        "recall": recall_ht
    }
    with open(METRICS_PATH, 'w') as f:
        f.write(str(metrics))

    return metrics 


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    tuning_data(X_train, y_train, X_test, y_test)