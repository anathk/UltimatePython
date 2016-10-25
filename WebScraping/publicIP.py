import requests, json

data = json.loads(requests.get("http://ip.jsontest.com/").text)
print(data["ip"])
