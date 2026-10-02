import fitz


def extract_text_from_pdf(pdf_file):
    documents = []

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text()

        if text.strip():
            documents.append({
                "text": text,
                "page": page_number
            })

    pdf.close()

    return documents