import requests

url = "http://localhost:9696/predict"

img_url = "./Capstone1_project/images/008.jpg"
img = {img_url} 

response = requests.post(url, json = img).json()
print(response)