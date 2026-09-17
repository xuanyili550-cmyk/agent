import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
_=load_dotenv()

from duckduckgo_search import DDGS
ddg=DDGS()
results=ddg.text("英伟达新的 Blackwell GPU 有什么特点？",max_results=4)
for r in results:
    print("-", r["title"])
    print("  ", r["body"][:200])

city="北京"
query=f"""
    {city}今天的天气怎么样？
    我今天适合出门？
    "weather.com"
"""
ddg=DDGS()
def search(query,max_result=6):
    try:
        results=ddg.text(query,max_results=max_result)
        return [i ["href"] for i in results]
    except Exception as e:
        results = [
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results
for i in search(query):
    print(i)

def scrape_weather_info(url):
    if not  url :
        return "Weather information could not be found."
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

url=search(query)[0]
soup=scrape_weather_info(url)
if isinstance(soup, str):   # 抓取失败时函数返回的是错误字符串，兜底成空文档，避免后面 soup.body / find_all 崩
    print(soup)
    soup=BeautifulSoup("", "html.parser")
print(f"Website: {url}\n\n")
print(str(soup.body)[:50000])

weather_data=[]
for tag in soup.find_all(['h1','h2','h3','p']):
    text=tag.get_text(' ',strip=True)
    weather_data.append(text)
weather_data="\n".join(weather_data)
weather_data=re.sub(r'\s+',' ',weather_data)
print(f"Website: {url}\n\n")
print(weather_data)
results=ddg.text(query,max_results=1)
data=results[0]['body'] if results else "未搜索到结果"
print(data)

import json
from pygments import highlight, lexers, formatters

results = ddg.text(query, max_results=3)
formatted_json=json.dumps(results,indent=4,ensure_ascii=True)
colorful_json=highlight(formatted_json,lexers.JsonLexer(),formatters.TerminalFormatter())
print(colorful_json)



client=TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
result=client.search("What is in Nvidia's new Blackwell GPU?",include_answer=True)
print(result["answer"])
city = "San Francisco"
query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""
def search(query,max_results=6):
    try:
        results=ddg.text(query,max_results=max_results)
        return [i ["href"] for i in results]
    except Exception as e:
        print(f"returning previous results due to exception reaching ddg.")
        results = [
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results
for i in search(query):
    print(i)

def scrape_weather_info(url):
    if not url:
        return "Weather information could not be found."
    headers={'User-Agent': 'Mozilla/5.0'}
    response=requests.get(url,headers=headers)
    if response.status_code!=200:
        return "Failed to retrieve the webpage."
    soup=BeautifulSoup(response.text,'html.parser')
    return soup
url = search(query)[0]
soup = scrape_weather_info(url)
if isinstance(soup, str):   # 同上：抓取失败兜底成空文档
    print(soup)
    soup = BeautifulSoup("", "html.parser")
print(f"Website: {url}\n\n")
print(str(soup.body)[:50000])
weather_data=[]
for tag in soup.find_all(['h1','h2','h3','p']):
    text=tag.get_text(" ",strip=True)
    weather_data.append(text)
weather_data="\n".join(weather_data)
weather_data=re.sub(r'\s+',' ',weather_data)
print(f"Website: {url}\n\n")
print(weather_data)

result=client.search(query,max_results=1)
data=result["results"][0]['content']
print(data)

try:
    parsed_json=json.loads(data.replace("'",'"'))
    formatted_json = json.dumps(parsed_json, indent=4)
except (json.JSONDecodeError, TypeError):
    formatted_json = data   # data 不是合法 JSON 时原样展示，避免 JSONDecodeError
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())

print(colorful_json)