import requests
from bs4 import BeautifulSoup
import json
import time

SPECIFIC_CHAPTERS = [
    "كتاب الطهارة"
    "كِتابُ الصَّلاةِ",
    "كتابُ الزَّكاةِ",
    "كتابُ الصَّوم",
    "كتابُ الحَجِّ"
]

def fetch_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            content_div = soup.find(class_="w-100 mt-4")
            if content_div:
                return content_div.text.strip()
        
    except Exception as e:
     pass
    return ""

def parse_list_item(li, level=1):

    node = {
        "level": level, 
        "text": "",
        "href": "",
        "children": [],
        "content": ""  
    }
    
    link = li.find("a", recursive=False)
    if link:
        node['text'] = link.text.strip()
        href = link.get("href")
        if href and href != "#":
            node['href'] = BASE_URL + href

    children = li.find("ul", recursive=False)
    if children:
        node['children'] = [parse_list_item(child, level=level + 1) for child in children.find_all("li", recursive=False)]
    else:
        if node['href']:
            node['content'] = fetch_content(node['href'])
            print(f"Fetched content for: {node['text']}")
           
    
    return node


BASE_URL = "https://dorar.net"
FEQHIA_URL = "https://dorar.net/feqhia"
response = requests.get(FEQHIA_URL)
if response.status_code != 200:
    print("Failed to fetch website content")
    exit()

soup = BeautifulSoup(response.text, "html.parser")
root_elements = soup.find_all(class_="mtree-node mtree-closed")

specific_elements = [
    element for element in root_elements
    if any(chapter in element.text for chapter in SPECIFIC_CHAPTERS)
]

tree = [parse_list_item(element, level=1) for element in specific_elements]

OUTPUT_JSON = "dorar.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tree, f, ensure_ascii=False, indent=4)

print(f"Tree for specific chapters successfully saved to {OUTPUT_JSON}")
