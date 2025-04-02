import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sys

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Separate features and target
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Define column types
    numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Create pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform the data
    X_processed = pipeline.fit_transform(X)

    return X_processed, y, pipeline

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_data_path = sys.argv[2]
    output_target_path = sys.argv[3]
    output_pipeline_path = sys.argv[4]

    df = load_data(input_path)
    X_processed, y, pipeline = preprocess_data(df)

    # Save processed data and pipeline
    pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed).to_csv(output_data_path, index=False)
    y.to_csv(output_target_path, index=False)
    joblib.dump(pipeline, output_pipeline_path)