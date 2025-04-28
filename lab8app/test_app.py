import requests

url = "http://127.0.0.1:8000/predict/"
payload = {
    "feature1": 0.5, "feature2": 1.2, "feature3": 2.1, "feature4": 3.5,
    "feature5": 4.0, "feature6": 5.6, "feature7": 6.1, "feature8": 7.7,
    "feature9": 8.2, "feature10": 9.9, "feature11": 10.0, "feature12": 11.5,
    "feature13": 12.3
}

response = requests.post(url, json=payload)

print(response.json())