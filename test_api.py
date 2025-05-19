import requests

url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}
payload = {
    "columns": ["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5", "pc2_1", "pc2_2"],
    "data": [[-0.5879220051210035, 0.06578591953724078, -0.24048954079238286,
              -0.07435319246219169, 0.1710674012484883,
              -0.029662744637863225, 0.14160488897853118]]
}

response = requests.post(url, json=payload, headers=headers)
print("Prediction:", response.json())
