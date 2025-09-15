def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"✅ Successfully read TXT file: {file_path}")
        print(f"📄 Characters extracted: {len(content)}")
        return content
    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
        return ""
    except Exception as e:
        print(f"❌ Error reading TXT: {e}")
        return ""

if __name__ == "__main__":
    text_content = read_txt("mohan_resume.txt.txt")
    print("TXT Content Preview:")
    print(text_content[:200] + "..." if len(text_content) > 200 else text_content)