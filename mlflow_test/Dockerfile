# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Install MLFlow and any dependencies
RUN pip install --no-cache-dir mlflow

# Expose the default port for MLFlow server
EXPOSE 5000

# Run MLFlow server
CMD ["mlflow", "ui", "--host", "0.0.0.0"]