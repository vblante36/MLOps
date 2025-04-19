from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib

def train_model(X_train, y_train, n_estimators=100, seed=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def register_model(model, accuracy, n_estimators, experiment="TrainingFlow"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "models/rf_model.pkl")


def load_registered_model(run_id=None):
    model_uri = f"runs:/{run_id}/model" if run_id else "models:/model/1"
    return mlflow.sklearn.load_model(model_uri)
