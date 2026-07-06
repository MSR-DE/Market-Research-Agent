import os 
import requests
from dotenv import load_dotenv

load_dotenv() ## reads env file

api_key = os.getenv("NEWS_API_KEY")
url = "https://newsapi.org/v2/everything"  ## use the top headline one later once this one's finished

def fetch_news(query):
    params = {
    "q": query,
    "apiKey": api_key,
    "pageSize": 5,               ## number of results per page
    "language": "en",
    "sortBy": "publishedAt",   
    "searchIn": "title,description" 
}
 
    response = requests.get(url, params=params)
    data = response.json()

    for article in data["articles"]:
        print(article["title"], "-", article["source"]["name"]) 


if __name__ == "__main__":
    fetch_news("Apple")
