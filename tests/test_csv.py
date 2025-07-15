from typing import List
import csv
import requests


def main(csv_path: str, target_col: int = 0, source_col: int = 1):
    target_texts = []
    source_texts = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader: 
            target_texts.append(row[target_col])
            source_texts.append(row[source_col])
    
    similarities = get_similarity(target_texts, source_texts)
    with open('./tests/output.csv', mode="w", newline="", encoding="utf-8") as new_file:
        writer = csv.writer(new_file)
        for i in range(0, len(target_texts)):
            writer.writerow([ target_texts[i], source_texts[i], similarities[i] ])


def get_similarity(texts1: List[str], texts2: List[str]):
    response = requests.post("http://localhost:8000/api/similarity", json={ 
        "texts1": texts1,
        "texts2": texts2,
    })

    response_body = response.json()
    similarities = list(map(lambda i: i['similarity'], response_body))
    return similarities



if __name__ == "__main__":
    main('./tests/input.csv')


