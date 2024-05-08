import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000")


with mlflow.start_run(run_name='xgboost_experiment_run') as run:
    # Load the diabetes dataset.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    print(predictions)

    mlflow.sklearn.log_model(rf, "model")
    mlflow.log_artifacts('/Users/shine/Documents/projects/finetune_llama3_with_ORPO/artifacts/model.pkl')

    print("Run ID: {}".format(run.info.run_id))