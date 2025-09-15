import docx

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        content = "\n".join([p.text for p in doc.paragraphs])
        print(f"✅ Successfully read DOCX file: {file_path}")
        print(f"📄 Characters extracted: {len(content)}")
        return content
    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
        return ""
    except Exception as e:
        print(f"❌ Error reading DOCX: {e}")
        return ""

if __name__ == "__main__":
    docx_content = read_docx("mohan_resume.docx.docx")
    print("DOCX Content Preview:")
    print(docx_content[:200] + "..." if len(docx_content) > 200 else docx_content)