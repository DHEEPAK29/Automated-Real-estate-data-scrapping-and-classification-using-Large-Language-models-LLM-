#loading Hugging Face api key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "d"

llm = HuggingFaceHub(
    repo_id="mistral-community/Mixtral-8x22B-v0.1",
    model_kwargs={"temperature":0.8, "max_length":100}
)
loaders = UnstructuredURLLoader(urls=[
    "https://webscraper.io/test-sites/e-commerce/allinone"
])
data = loaders.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)
query = "just get the data. Organize and write it"
langchain.debug=True
try:
  result = chain({"query": query}, return_only_outputs=False)
except:
  print("err")


print(result)
text = "Toggle navigation\n\nWeb Scraper\n\nCloud Scraper\n\nPricing\n\nLearn\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\tDocumentation\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\tVideo Tutorials\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\tHow to\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\tTest Sites\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\tForum\n\nInstall\n\nCloud Login\n\nTest Sites\n\nHome\n\nComputers\n\nPhones\n\nE-commerce training site\n\nWelcome to WebScraper e-commerce site. You can use this site for training\n\t\t\tto learn how to use the Web Scraper. Items listed here are not for sale.\n\nTop items being scraped right now\n\n$1096.02\n\nLenovo ThinkPa...\n\nLenovo ThinkPad L460, 14\" FHD IPS, Core i7-6600U, 8GB, 256GB SSD, Windows 10 Pro\n\n14 reviews\n\n$1203.41\n\nToshiba Porteg...\n\nToshiba Portege X30-D-10J Black/Blue, 13.3\" FHD IPS, Core i5-7200U, 8GB, 256GB SSD, Windows 10 Pro\n\n13 reviews\n\n$1101.83\n\nAsus ROG Strix...\n\nApple MacBook Air 13.3\", Core i5 1.8GHz, 8GB, 128GB SSD, Intel HD 4000, RUS\n\n4 reviews\n\nProducts\n\nWeb Scraper browser extension\n\nWeb Scraper Cloud\n\nCompany\n\nAbout us\n\nContact\n\nAsus ROG Strix...\n\nApple MacBook Air 13.3\", Core i5 1.8GHz, 8GB, 128GB SSD, Intel HD 4000, RUS\n\n4 reviews\n\nProducts\n\nWeb Scraper browser extension\n\nWeb Scraper Cloud\n\nCompany\n\nAbout us\n\nContact\n\nWebsite Privacy Policy\n\nBrowser Extension Privacy Policy\n\nMedia kit\n\nJobs\n\nResources\n\nBlog\n\nDocumentation\n\nVideo Tutorials\n\nScreenshots\n\nTest Sites\n\nForum\n\nStatus\n\nCONTACT US\n\ninfo@webscraper.io\n\nUbelu 5-71, Adazi, Latvia, LV-2164\n\n\n\n\n\n\n\n\n\n\n\nCopyright &copy 2024\n\t\t\t\t\tWeb Scraper | All rights\n\t\t\t\t\treserved"
# text = result['result']
answer_start = text.find("context")

# Extract the text after "Answer:" until a blank line is found
answer_text = text[answer_start:text.find("\n\n", answer_start)]

print(answer_text)



import requests
import os

# Use your personal token here
TOKEN = "-_-" #os.environ.get("API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "The answer to the universe is",
})
print(headers)
print(output)

