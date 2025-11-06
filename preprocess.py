import re

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = re.sub(r"http\S+|www\S+", "", text)  # remove links
    return text.strip()  # remove extra spaces

# Example test
if __name__ == "__main__":
    print(preprocess_text("ML is <fun>"))  # output: "ml is"