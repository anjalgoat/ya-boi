import requests

def find_appstore_url(app_name, country='us'):
    url = "https://itunes.apple.com/search"
    params = {
        "term": app_name,
        "entity": "software",
        "limit": 5,
        "country": country
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['resultCount'] > 0:
        for result in data['results']:
            print(f"App: {result['trackName']}")
            print(f"URL: {result['trackViewUrl']}\n")
    else:
        print("No results found.")

# Example usage
find_appstore_url("dream leauge soccer")
