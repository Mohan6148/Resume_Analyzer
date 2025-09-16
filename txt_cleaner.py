import re

def clean_text(text):
    # Remove extra whitespace and empty lines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # You can add more cleaning steps as needed
    return text

if __name__ == "__main__":
    example = "  This   is an   EXAMPLE   text.\n\nNext line. "
    print(clean_text(example))