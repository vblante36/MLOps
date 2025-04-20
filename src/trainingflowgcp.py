from metaflow import FlowSpec, step, kubernetes, resources, retry, timeout, catch, conda_base, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

@conda_base(python="3.10")
class TrainingFlowGCP(FlowSpec):
    """
    A Metaflow training pipeline that:
    1. Loads and splits data
    2. Preprocesses features
    3. Trains a model
    4. Evaluates performance
    5. Logs and registers the model with MLFlow
    """

    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)

    @step
    def start(self):
        print("Loading data...")
        self.data = pd.read_csv('/Users/victoriablante/1-MSDS/Spring Mod 2/MLOps/data/heart.csv')
        self.next(self.preprocess)

    @step
    def preprocess(self):
        print("Splitting data...")
        X = self.data.drop('HeartDisease', axis=1)
        y = self.data['HeartDisease']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)
        self.next(self.train_model)

    @step
    @kubernetes
    @resources(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=300)
    @catch(var="error")
    def train_model(self):
        print("Training model with preprocessing...")

        cat_cols = self.X_train.select_dtypes(include='object').columns.tolist()
        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])

        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', RandomForestClassifier(random_state=self.seed))
        ])

        self.pipeline.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @step
    @retry(times=2)
    @timeout(seconds=180)
    @catch(var="error")
    def evaluate(self):
        print("Evaluating...")
        preds = self.pipeline.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, preds)
        print("Accuracy:", self.accuracy)
        self.next(self.register_model)

    @step
    @retry(times=2)
    @timeout(seconds=180)
    @catch(var="error")
    def register_model(self):
        print("Logging model to MLFlow...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Or your remote URI
        mlflow.set_experiment("Heart Model")

        with mlflow.start_run():
            mlflow.log_param("seed", self.seed)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(
                self.pipeline,
                "model",
                registered_model_name="HeartClassifier"
            )
        self.next(self.end)

    @step
    def end(self):
        print("Training complete and model registered.")

if __name__ == '__main__':
    TrainingFlowGCP()