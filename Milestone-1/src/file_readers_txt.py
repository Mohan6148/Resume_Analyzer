def read_txt(file_bytes):
    """Extract text from TXT file bytes (for Streamlit uploads)."""
    try:
        content = file_bytes.decode('utf-8')
        return content
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return ""