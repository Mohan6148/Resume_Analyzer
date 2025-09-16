from file_readers_docx import read_docx
from file_readers_pdf import read_pdf
from file_readers_txt import read_txt
from txt_cleaner import clean_text
from remove_personal import remove_personal_info

def process_resume(file_path, file_type):
    if file_type == 'docx':
        text = read_docx(file_path)
    elif file_type == 'pdf':
        text = read_pdf(file_path)
    elif file_type == 'txt':
        text = read_txt(file_path)
    else:
        print("Unsupported file type.")
        return

    text = clean_text(text)
    text = remove_personal_info(text)
    print("Processed Resume Content:")
    print(text[:200] + "..." if len(text) > 200 else text)

if __name__ == "__main__":
    # Example: change as needed
    process_resume("mohan_resume.pdf.pdf", "pdf")