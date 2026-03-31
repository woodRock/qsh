#!/usr/bin/env python3
import requests
url = "http://localhost:8080/completion"
data = {"prompt": "list files", "n_predict": 20}
r = requests.post(url, json=data)
print(r.json())
