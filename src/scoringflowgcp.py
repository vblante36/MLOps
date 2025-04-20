from metaflow import FlowSpec, step, kubernetes, resources, retry, timeout, catch, conda_base
import pandas as pd
import mlflow
import mlflow.sklearn

@conda_base(python="3.10")
class ScoringFlowGCP(FlowSpec):
    """
    A Metaflow scoring pipeline that:
    1. Loads new data
    2. Loads the registered MLFlow model
    3. Makes predictions
    """

    @step
    def start(self):
        print("Loading new data...")
        self.data = pd.read_csv('/Users/victoriablante/1-MSDS/Spring Mod 2/MLOps/data/heart.csv')  # Replace with new data if available
        self.X = self.data.drop('HeartDisease', axis=1)
        self.next(self.load_model)

    @step
    @kubernetes
    @resources(cpu=2, memory=2048)
    @retry(times=2)
    @timeout(seconds=180)
    @catch(var="error")
    def load_model(self):
        print("Loading model from MLFlow...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Or remote URI
        model_uri = "models:/HeartClassifier/Production"
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    @retry(times=2)
    @timeout(seconds=180)
    @catch(var="error")
    def predict(self):
        print("Making predictions...")
        self.preds = self.pipeline.predict(self.X)
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")
        print("Sample predictions:", self.preds[:5])

if __name__ == '__main__':
    ScoringFlowGCP()