import mlflow
import mlflow.sklearn
import pandas as pd

def log_experiment(args, accuracy:float, report:dict, model):

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)

        mlflow.log_metric("accuracy", accuracy)

        report_df = pd.DataFrame(report).transpose()
        report_path = "classification_report.csv"
        report_df.to_csv(report_path, index=True)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(model, "model")

