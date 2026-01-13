import os
import hashlib
import re
from pathlib import Path
from tqdm import tqdm
from typing import List
from docling_core.types.doc import PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Directory paths
DATA_DIR = "data/pdfs"
OUTPUT_MD_DIR = "data/markdown"
OUTPUT_FIGURES_DIR = "data/figures"
OUTPUT_DESCRIPTIONS_DIR = "data/figures_descriptions"
OUTPUT_TABLES_DIR = "data/tables"

MODEL_NAME = ""
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = ""

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_MD_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DESCRIPTIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)

def convert_pdf_to_docling(pdf_file: Path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale=3.0
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return doc_converter.convert(pdf_file)

def save_page_images(doc_converter, figures_dir: Path):
    pages_to_save = set()

    for item in doc_converter.document.iterate_items():
        element = item[0]

        if isinstance(element, PictureItem):
            image = element.get_image(doc_converter.document)

            if image.size[0] > 500 or image.size[1] > 500:
                page_no = element.prov[0].page_no if element.prov else None

                if page_no:
                    pages_to_save.add(page_no)

        for page_no in pages_to_save:
            page = doc_converter.document.get_page[page_no]

            page.image.pil_image.save(figures_dir / f"{page_no}.png", "PNG")

def extract_context_and_table(lines: List[str], table_index: int):
    table_lines = []
    i = table_index

    while (i < len(lines)) and (lines[i].startswith("|")):
        table_lines.append(lines[i])
        i += 1

    start = max(0, table_index - 2)
    context_lines = lines[start: table_index]

    content = "\n".join(context_lines) + "\n\n" + "\n".join(table_lines)

    return content, i

def extract_tables_with_content(markdown_text: str):
    lines = markdown_text.split("\n")
    lines = [line for line in lines if line.strip()]
    tables = []
    current_page = 1
    table_num = 1
    i = 0

    while(i < len(lines)):
        if '<!-- page break -->' in lines[i]:
            current_page = current_page + 1
            i = i + 1
            continue

        if lines[i].startswith("|") and lines[i].count("|")>1:
            content, next_i = extract_context_and_table(lines, i)
            tables.append((content, f"table_{table_num}", current_page))
            table_num = table_num + 1
            i = next_i
        else:
            i = i + 1

    return lines

def save_tables(markdown_text, tables_dir):

    tables = extract_tables_with_content(markdown_text)

    for table_context, table_name, page_num in tables:
        context_with_page = f"**Page:** {page_num}\n\n{table_context}"

        (tables_dir / f"{table_name}_page_{page_num}.md").write_text(context_with_page, encoding="utf-8")

def extract_pdf_content(pdf_file):
    md_dir = Path(OUTPUT_MD_DIR) / pdf_file.stem
    figures_dir = Path(OUTPUT_FIGURES_DIR) / pdf_file.stem
    tables_dir = Path(OUTPUT_TABLES_DIR) / pdf_file.stem

    for dir_path in [md_dir, figures_dir, tables_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    doc_converter = convert_pdf_to_docling(pdf_file)
    markdown_text = doc_converter.document.export_to_markdown(page_break_placeholder="<!-- page_break -->")

    (md_dir / f"{pdf_file.stem}.md").mkdir(parents=True, exist_ok=True)

    save_page_images(doc_converter, figures_dir)

    save_tables(markdown_text, tables_dir)

def compute_file_hash(file_path: Path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()

def get_processed_hashes():
    processed_hashes = set()
    offset = None

    while True:
        points, offset = vector_store.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10_000,
            with_payload=True,
            offset=offset
        )

        if not points:
            break

        processed_hashes.update(point.payload['metadata']['file_hash'] for point in points)

        if offset is None:
            break
    return processed_hashes

def extract_page_number(file_path: Path):
    pattern = r'page_(\d+)'
    match = re.search(pattern=pattern, string=file_path.stem)
    return int(match.group(1)) if match else None


''' embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL) '''
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

vector_store = QdrantVectorStore.from_douments(
    documents=[],
    embeddings=embeddings,
    sparse_embeddings=sparse_embeddings,
    url="http://localhost:6333",
    collection_name = COLLECTION_NAME,
    retrieval_mode=RetrievalMode.HYBRID,
    force_recreate=False
)

processed_hashes = get_processed_hashes()
def ingest_file_in_db(file_path, processed_hashes):

    file_hash = compute_file_hash(file_path)
    if file_hash in processed_hashes:
        print(f"Following file has been already uploaded: {file_path}")

    path_str = str(file_path)
    if 'markdown' in path_str:
        content_type = 'text'
        doc_name = file_path.name
    elif 'tables' in path_str:
        content_type = 'tables'
        doc_name = file_path.parent.name
    elif 'image_desc' in path_str:
        content_type = 'image_desc'
        doc_name = file_path.parent_name
    else:
        content_type = 'unknown'
        doc_name = file_path.name

    content = file_path.read_text(encoding="utf-8")


    base_metadata = {
        'content_type': content_type,
        'file_hash': file_hash,
        'source_file': doc_name,
    }

    if content_type == 'text':
        pages = content.split("<!-- page break -->")
        documents = []
        for idx, page in enumerate(pages, start=1):
            metadata = base_metadata.copy()
            metadata.update({'page': idx})
            documents.append(Document(page_content=page, metadata=base_metadata))

        vector_store.add_documents(documents)
    else:
        page_num = extract_page_number(file_path)
        metadata = base_metadata.copy()
        metadata.update({'page': page_num})
        documents = [Document(page_content=content, metadata=base_metadata)]

        vector_store.add_documents(documents)

    processed_hashes.add(file_hash)


data_path = Path(DATA_DIR)
pdf_files = data_path.glob("*.pdf")

for idx, pdf_file in enumerate(pdf_files):
    print(pdf_file)
    extract_pdf_content(pdf_file)


base_path = Path(OUTPUT_MD_DIR)
all_md_files = list(base_path.rglob("*.md"))

for md_file in tqdm(all_md_files):
    ingest_file_in_db(md_file, processed_hashes)

collection_info = vector_store.client.get_collection_info(COLLECTION_NAME)