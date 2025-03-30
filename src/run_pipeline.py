import argparse
from data_loader import load_data
from preprocessing import preprocess_data, split_data
from training import train_model
from evaluation import evaluate_model
from mlflow_logger import log_experiment

def main(args):

    df = load_data()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df, test_size= args.test_size, random_state= args.random_state)

    model = train_model(X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)

    accuracy, report = evaluate_model(model, X_test, y_test)

    log_experiment(args, accuracy, report, model)

    print(f"Model logged with accuracy: {accuracy:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ML Pipeline with MLFlow Tracking")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximumm depth of each tree" )   
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")

    args = parser.parse_args()
    main(args)