import os
import fitz  # PyMuPDF for PDFs
import tempfile
from PIL import Image
import docx
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import logging
from pymongo import MongoClient
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the language model
llm = ChatGroq(
    groq_api_key='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',  # Replace with your actual API key
    model_name = "llama-3.3-70b-specdec",
    temperature=0.2,
    max_tokens=3000,
    model_kwargs={"top_p": 1}
)

# Define the chat prompt template
template = """
<|context|>
You are a medical AI that helps users find online purchase links for their prescribed medications. 

Your task:
1. Extract the **medication names** from the uploaded prescription.
2. Search for **reliable online pharmacy links** (example: 1mg, Netmeds, PharmEasy).
3. Present the medications **along with their links** in a **simple format**.
4. If a medication is not found, suggest an alternative brand.

</s>
<|user|>
Prescription data: 
{report_text}

</s>
<|assistant|>
"""



prompt = ChatPromptTemplate.from_template(template)

    # Function to pass text to LLM for analysis
def format_response_with_links(response_text):
    response_text = response_text.replace("https://", '<a href="https://').replace("\n", '">\n').replace(" ", '" target="_blank">')  
    return response_text

def analyze_text_with_llm(report_text):
    try:
        # Format the prompt
        formatted_prompt = prompt.format(report_text=report_text)
        # Directly pass formatted prompt to LLM (without using to_input_dict)
        logging.info(f"Formatted Prompt Sent to LLM: {formatted_prompt}")
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        logging.error(f"Error while processing with LLM: {e}")
        return "Error: Unable to analyze the report."

# Function to process image files
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to process PDF files using fitz
def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)  # Save the uploaded file to the temporary file
        pdf_file = fitz.open(temp_file.name)
        text = ""
        for page_num in range(pdf_file.page_count):
            page = pdf_file.load_page(page_num)
            text += page.get_text()
    return text

# Function to process Word documents
def extract_text_from_word(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Main function to analyze report based on file type
def analyze_report(file):
    file_extension = file.filename.split('.')[-1].lower()

    # Extract text based on file type
    if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
        extracted_text = extract_text_from_image(file)
    elif file_extension == 'pdf':
        extracted_text = extract_text_from_pdf(file)
    elif file_extension in ['doc', 'docx']:
        extracted_text = extract_text_from_word(file)
    else:
        return "Unsupported file type."



    # Analyze extracted text with LLM
    return analyze_text_with_llm(extracted_text)
