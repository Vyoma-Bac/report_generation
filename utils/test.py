from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfWriter, PdfReader
from io import BytesIO

# Step 1: Generate the main PDF and save it to a buffer
main_pdf_buffer = BytesIO()

def generate_main_pdf(pdf_buffer):
    main_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)
    for i in range(1, 6):  # Creating 5 pages
        main_canvas.drawString(100, 750, f"This is page {i}.")
        main_canvas.showPage()
    main_canvas.save()

generate_main_pdf(main_pdf_buffer)

# Move the buffer's position to the beginning
main_pdf_buffer.seek(0)

# Step 2: Create the new first page and save it to a buffer
first_page_buffer = BytesIO()

def create_first_page(pdf_buffer):
    first_page_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)
    first_page_canvas.drawString(100, 750, "This is the new first page.")
    first_page_canvas.save()

create_first_page(first_page_buffer)

# Move the buffer's position to the beginning
first_page_buffer.seek(0)

# Step 3: Merge the new first page with the existing PDF and save it to a buffer
final_pdf_buffer = BytesIO()

def merge_pdfs(first_page_buffer, main_pdf_buffer, output_buffer):
    output = PdfWriter()

    # Read the new first page
    first_page_pdf = PdfReader(first_page_buffer)
    output.add_page(first_page_pdf.pages[0])

    # Read the original PDF and add its pages
    main_pdf = PdfReader(main_pdf_buffer)
    for page_num in range(len(main_pdf.pages)):
        output.add_page(main_pdf.pages[page_num])

    # Save the final PDF to the buffer
    output.write(output_buffer)

merge_pdfs(first_page_buffer, main_pdf_buffer, final_pdf_buffer)

# Move the buffer's position to the beginning
final_pdf_buffer.seek(0)

# Save the final PDF buffer to a file (for demonstration purposes)
with open("final_example.pdf", "wb") as f:
    f.write(final_pdf_buffer.getvalue())

print("PDF generation and merging complete. The final PDF is saved as 'final_example.pdf'.")
