import pdfplumber

def read_pdf(file_path):
    content = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                content += page.extract_text() or ""
        return content.strip()
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""

if __name__ == "__main__":
    pdf_content = read_pdf("mohan_resume.pdf.pdf")
    print("PDF Content Preview:")
    print(pdf_content[:200] + "..." if len(pdf_content) > 200 else pdf_content)
