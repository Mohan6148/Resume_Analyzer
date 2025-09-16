import docx

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        content = "\n".join([p.text for p in doc.paragraphs])
        return content
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""

# Optional standalone run for quick test
if __name__ == "__main__":
    docx_content = read_docx("mohan_resume.docx.docx")
    print("DOCX Content Preview:")
    print(docx_content[:200] + "..." if len(docx_content) > 200 else docx_content)
