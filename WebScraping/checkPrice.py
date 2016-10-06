import requests
from bs4 import BeautifulSoup

session = requests.Session()

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5)AppleWebKit 537.36 (KHTML, like Gecko) Chrome",\
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"}


url = 'http://www.bestbuy.com/site/sony-alpha-a7-ii-full-frame-mirrorless-camera-with-28-70mm-lens-black/1634012.p?skuId=1634012'
url1 = 'http://www.bestbuy.com'

req = session.get(url, headers=headers)

soup = BeautifulSoup(req.text, "html.parser")

print(soup.find("div", class_="item-price").get_text())

#print(soup.prettify())

print(soup.find("span", class_="price").get_text())