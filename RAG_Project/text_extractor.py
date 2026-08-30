import re

import requests
from bs4 import BeautifulSoup


def extract_main_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    response = requests.get(url, timeout=20, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        raise ValueError("Could not find the main article content on the page.")

    paragraphs = content.find_all("p")
    text_parts = []

    for p in paragraphs:
        clean_text = " ".join(p.get_text(separator=" ", strip=True).split())
        if clean_text:
            text_parts.append(clean_text)

    return "\n\n".join(text_parts)


def main():
    url = "https://en.wikipedia.org/wiki/Golden_Retriever"
    output_path = "Selected_Document.txt"

    article_text = extract_main_text(url)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(article_text)

    print(f"Saved article text to {output_path}")


if __name__ == "__main__":
    main()
