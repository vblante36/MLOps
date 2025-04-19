from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class TrainingFlow(FlowSpec):
    """
    A Metaflow training pipeline that:
    1. Loads and splits data
    2. Preprocesses features
    3. Trains a model
    4. Evaluates performance
    5. Logs and registers the model with MLFlow
    """

    # Parameters to allow reproducibility and control over data split
    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)

    @step
    def start(self):
        """
        Step 1: Load data from CSV into a DataFrame.
        """
        print("Loading data...")
        self.data = pd.read_csv('/Users/victoriablante/1-MSDS/Spring Mod 2/MLOps/data/heart.csv')
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """
        Step 2: Split data into training and test sets.
        """
        print("Splitting data...")
        X = self.data.drop('HeartDisease', axis=1)
        y = self.data['HeartDisease']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Step 3: Build a pipeline with preprocessing and a Random Forest model,
        then train it on the training data.
        """
        print("Training model with preprocessing...")

        # Separate numeric and categorical features
        cat_cols = self.X_train.select_dtypes(include='object').columns.tolist()
        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Define preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])

        # Combine preprocessing with classifier
        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', RandomForestClassifier(random_state=self.seed))
        ])

        # Fit pipeline on training data
        self.pipeline.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """
        Step 4: Evaluate the trained model on the test data.
        """
        print("Evaluating...")
        preds = self.pipeline.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, preds)
        print("Accuracy:", self.accuracy)
        self.next(self.register_model)

    @step
    def register_model(self):
        """
        Step 5: Log the model to MLFlow and register it.
        """
        print("Logging model to MLFlow...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Or replace with cloud tracking URI
        mlflow.set_experiment("Heart Model")
        
        # Log parameters, metrics, and model itself
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
        """
        Final step: Confirm training and registration are complete.
        """
        print("Training complete and model registered.")

# Entry point
if __name__ == '__main__':
    TrainingFlow()