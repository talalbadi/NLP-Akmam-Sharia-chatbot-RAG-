import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://ifta.ly"

def fetch_question_and_answer(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("h1", class_="entry-title").text.strip() if soup.find("h1", class_="entry-title") else "No question found"
            answer_div = soup.find("div", class_="entry-content")
            question =answer_div.findAll("p")[3].text.strip()
            for i in answer_div.findAll("p")[4:]:
                if("الجواب"  in i.text.strip()):
                    break
                question +=i.text.strip()
            
            answer = answer_div.text.strip() if answer_div else "No answer found"

            return {"title":title,"question": question, "answer": answer}
    except Exception as e:
        print(f"{url} - {e}")
    return {"question": "Error fetching question", "answer": "Error fetching answer"}

def scrape_questions(base_url):
    questions_and_answers = []
    for i in range(58):
      try:
          response = requests.get(base_url+f"page/{i}/")
          if response.status_code != 200:
             print(f"Failed to fetch {base_url}")
             return []

          soup = BeautifulSoup(response.text, "html.parser")

          question_links = soup.find_all(class_="post")

          for article in question_links:
            link = article.find("a")["href"]
            if link:
                full_url = link if link.startswith("http") else BASE_URL + link
                print(f"Fetching question and answer from: {full_url}")
                qa_pair = fetch_question_and_answer(full_url)
                questions_and_answers.append(qa_pair)
                

      except Exception as e:
         print(f"Error scraping questions: {e}")

    return questions_and_answers



URL = "https://ifta.ly/category/%d8%a7%d9%84%d9%81%d8%aa%d8%a7%d9%88%d9%89/%d8%a7%d9%84%d8%b9%d8%a8%d8%a7%d8%af%d8%a7%d8%aa/"


questions_and_answers = scrape_questions(URL)
OUTPUT_JSON = "questions_and_answers.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(questions_and_answers, f, ensure_ascii=False, indent=4)

print(f"Questions and answers successfully saved to {OUTPUT_JSON}")
