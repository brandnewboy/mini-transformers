import requests

data = requests.get('https://www.tp.xyz/')
print(data.text)