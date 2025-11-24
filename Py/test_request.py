import requests

url = "http://127.0.0.1:5000/diabetes/v1/predict"

data = {
    "Glucose": 120,
    "BMI": 30.1,
    "Age": 45
}

response = requests.post(url, json=data)
print(response.json())
