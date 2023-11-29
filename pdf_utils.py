from fpdf import FPDF


def generate_full_pdf(data_dict=None):
    pdf = FPDF()

    pdf.set_font("Arial", size=12)

    section_spacing = 10

    for key, value in data_dict.items():
        pdf.add_page()

        pdf.set_font("Arial", size=14, style="B")
        pdf.cell(0, 10, txt=key, ln=True)
        pdf.set_font("Arial", size=12, style="")

        if key == "Mind Map":
            pdf.image("mind_map.png", x=10, y=20, w=180)
        elif isinstance(value, str):
            paragraphs = value.split("\n\n")
            for paragraph in paragraphs:
                pdf.multi_cell(0, 10, txt=paragraph)
        elif isinstance(value, list) and value:
            if isinstance(value[0], list):
                value = ["\n".join(val) for val in value if isinstance(val, str)]
            value_text = "\n".join([val for val in value if isinstance(val, str)])
            pdf.multi_cell(0, 10, txt=value_text)
        else:
            pdf.multi_cell(0, 10, txt=str(value))

        pdf.ln(section_spacing)

    pdf_file_path = "Notes.pdf"

    pdf.output(pdf_file_path)

    return pdf_file_path


def generate_user_notes_pdf(text):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    page_width = pdf.w - 2 * pdf.l_margin

    paragraphs = text.split("\n\n")

    for i, paragraph in enumerate(paragraphs):
        lines = paragraph.split("\n")

        formatted_lines = []
        current_line = ""

        for line in lines:
            words = line.split()
            for word in words:
                if pdf.get_string_width(current_line + word) < page_width:
                    current_line += word + " "
                else:
                    formatted_lines.append(current_line)
                    current_line = word + " "
            formatted_lines.append(current_line.strip())
            current_line = ""

        for line in formatted_lines:
            pdf.cell(200, 10, txt=line, ln=True)

        if i < len(paragraphs) - 1:
            pdf.cell(200, 10, txt="", ln=True)

    pdf_file_path = "Notes.pdf"
    pdf.output(pdf_file_path)

    return pdf_file_path
