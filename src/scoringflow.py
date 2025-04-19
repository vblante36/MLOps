from metaflow import FlowSpec, step
import pandas as pd
import mlflow
import mlflow.sklearn

class ScoringFlow(FlowSpec):
    """
    A Metaflow scoring pipeline that:
    1. Loads new data
    2. Loads the registered MLFlow model
    3. Makes predictions
    """

    @step
    def start(self):
        """
        Step 1: Load new data.
        This example uses the same dataset as training, but ideally you use unseen data.
        """
        print("Loading new data...")
        self.data = pd.read_csv('/Users/victoriablante/1-MSDS/Spring Mod 2/MLOps/data/heart.csv')  # Replace with new data if available
        self.X = self.data.drop('HeartDisease', axis=1)  # Features only
        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Step 2: Load the MLFlow registered model from the Production stage.
        Assumes MLFlow is running locally.
        """
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLFlow URI
        model_uri = "models:/HeartClassifier/Production"  # Replace with correct model name/stage if different
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        """
        Step 3: Run predictions using the loaded pipeline on the new data.
        """
        print("Making predictions...")
        self.preds = self.pipeline.predict(self.X)
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Print a confirmation and sample predictions.
        """
        print("Scoring complete.")
        print("Sample predictions:", self.preds[:5])

# Entry point
if __name__ == '__main__':
    ScoringFlow()