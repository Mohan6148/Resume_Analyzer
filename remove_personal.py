import re

def remove_personal_info(text):
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Remove phone numbers (simple pattern, adjust as needed)
    text = re.sub(r'\b\d{10,13}\b', '[PHONE]', text)
    return text

if __name__ == "__main__":
    sample = "Contact me at john.doe@email.com or 9876543210"
    print(remove_personal_info(sample))
