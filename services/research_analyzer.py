import os
import fitz  # PyMuPDF
import base64
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import yake

# --- LOAD TRANSFORMER MODEL (Summarization) ---
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# --- TEXT EXTRACTION ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

# --- CLEAN TEXT ---
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    skip_prefixes = ['Table', 'Figure', 'Fig.', 'TABLE', 'FIGURE']
    for line in lines:
        if any(line.strip().startswith(prefix) for prefix in skip_prefixes):
            continue
        if len(line.strip()) < 5:
            continue
        cleaned.append(line.strip())
    return ' '.join(cleaned)

# --- SUMMARIZATION TASKS ---
def generate_summary(text, task="summary"):
    prompt_map = {
        "summary": "Summarize the research paper:",
        "methodology": "Explain the methodology used in this paper:",
        "risks": "What are the limitations or risks of this study?",
        "relevance": "Explain the clinical or practical relevance of the study:",
        "layman": "Explain this research to a high school student in simple terms:"
    }

    # Truncate to acceptable token length (~1024 for BigBird-compatible summarization head)
    input_text = prompt_map[task] + "\n\n" + text[:4000]
    result = summarizer(input_text, max_length=250, min_length=60, do_sample=False)
    return result[0]['summary_text']

# --- KEYWORDS ---
def extract_keywords(text, max_keywords=10):
    cleaned = clean_text(text)
    try:
        extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
        keywords = extractor.extract_keywords(cleaned)
        return [kw for kw, _ in keywords]
    except Exception as e:
        return [f"Keyword extraction error: {str(e)}"]

# --- IMAGE EXTRACTOR ---
def image_extractor(pdf_path, output_dir="extracted_images", return_base64=False):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted_images = []

    for page_index, page in enumerate(doc):
        try:
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_name = f"page{page_index+1}_img{img_index+1}.{image_ext}"
                    image_path = os.path.join(output_dir, image_name)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    image_info = {
                        "filename": image_name,
                        "ext": image_ext,
                        "page": page_index + 1,
                        "index": img_index + 1,
                    }

                    if return_base64:
                        image_info["base64"] = base64.b64encode(image_bytes).decode("utf-8")
                    else:
                        image_info["path"] = image_path

                    extracted_images.append(image_info)
                except Exception as e:
                    extracted_images.append({"error": f"Image extraction failed on page {page_index+1}, image {img_index+1}: {str(e)}"})
        except Exception as e:
            extracted_images.append({"error": f"Image extraction failed on page {page_index+1}: {str(e)}"})

    return extracted_images

# --- MAIN ANALYSIS ---
def analyze_research_paper(pdf_path):
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)

    return {
        "summary": generate_summary(cleaned, task="summary"),
        "methodology": generate_summary(cleaned, task="methodology"),
        "risks": generate_summary(cleaned, task="risks"),
        "clinical_relevance": generate_summary(cleaned, task="relevance"),
        "layman_translation": generate_summary(cleaned, task="layman"),
        "keywords": extract_keywords(cleaned),
        "images": image_extractor(pdf_path, return_base64=True)
    }
