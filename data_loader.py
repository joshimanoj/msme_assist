import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import logging
import gdown  # For downloading from Google Drive
import hashlib
import requests
from utils import get_embeddings

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["load_rag_data", "load_dfl_data"]

def load_rag_data(
    google_drive_file_id="13OKwV3G_j4OJdm1Cxt7FAlO6qzrxBMW6",
    faiss_index_path="faiss_index",
    version_file="faiss_version.txt",
    cached_file_path="scheme_db_latest.xlsx",
    chunk_size=300,
):
    """
    Load scheme_db.xlsx, check for precomputed FAISS index, and return a FAISS vector store.
    If a valid precomputed index exists, load it; otherwise, process in chunks and create a new index.
    
    Args:
        google_drive_file_id (str): The file ID from the Google Drive shareable link.
        faiss_index_path (str): Directory containing the precomputed FAISS index.
        version_file (str): File containing the hash of the Excel file used for the index.
    
    Returns:
        FAISS: FAISS vector store with indexed scheme documents.
    """
    # Check if index and version hash exist and cached file matches
    if os.path.exists(faiss_index_path) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if os.path.exists(cached_file_path):
                with open(cached_file_path, "rb") as cf:
                    cached_hash = hashlib.md5(cf.read()).hexdigest()
                if cached_hash == stored_hash:
                    logger.info(
                        f"Using existing FAISS index and cached file with hash {stored_hash}"
                    )
                    embeddings = get_embeddings()
                    return FAISS.load_local(
                        faiss_index_path,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
        except Exception as e:
            logger.error(
                f"Failed to load precomputed FAISS index: {str(e)}. Will attempt download."
            )

    # Download the Google Sheet as an Excel file
    download_url = f"https://docs.google.com/spreadsheets/d/{google_drive_file_id}/export?format=xlsx"
    temp_file_path = cached_file_path
    logger.info(
        f"Downloading Google Sheet from Google Drive (File ID: {google_drive_file_id}) to {temp_file_path}"
    )
    try:
        gdown.download(download_url, temp_file_path, quiet=False)
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Failed to download file from Google Drive: {str(e)}")
        raise

    # Compute file hash
    with open(temp_file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    logger.info(f"Computed file hash: {file_hash}")

    # If index exists and hash matches, skip processing
    if os.path.exists(faiss_index_path) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if stored_hash == file_hash:
                logger.info(
                    f"Precomputed FAISS index found with matching hash at {faiss_index_path}"
                )
                embeddings = get_embeddings()
                return FAISS.load_local(
                    faiss_index_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.info(
                    f"Hash mismatch: stored hash {stored_hash}, current hash {file_hash}. Recomputing FAISS index."
                )
        except Exception as e:
            logger.error(
                f"Failed to load precomputed FAISS index: {str(e)}. Recomputing FAISS index."
            )

    else:
        logger.info(
            f"No precomputed FAISS index found at {faiss_index_path} or version file missing. Computing new index."
        )



    # Read Excel file
    try:
        df = pd.read_excel(temp_file_path)
        logger.info(f"Excel file loaded successfully. Rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to read Excel file: {str(e)}")
        raise
    finally:
        # Keep the downloaded file for future reuse
        logger.info(f"Cached file available at {temp_file_path}")

    relevant_columns = [
        "Scheme GUID",
        "Scheme Name",
        "parent_scheme_name",
        "Applicability (State)",
        "Central Department Name",
        "State department name",
        "Type (Sch/Doc)",
        "Service Type Name",
        "Scheme description",
        "Scheme Eligibility",
        "Application Process",
        "Benefit"
    ]

    embeddings = get_embeddings()
    vector_store = None

    def process_rows(rows):
        docs = []
        for _, row in rows.iterrows():
            parts = []
            for col in relevant_columns:
                if col in row and pd.notna(row[col]):
                    clean_col = col.replace('(', ' ').replace(')', '')
                    parts.append(f"{clean_col}: {row[col]}")
            content = "\n".join(parts)
            metadata = {
                "guid": row.get("Scheme GUID", ""),
                "name": row.get("Scheme Name", ""),
                "applicability": row.get("Applicability (State)", ""),
                "type": row.get("Type (Sch/Doc)", "")
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    logger.info(f"Processing {len(df)} rows in chunks of {chunk_size}")
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        documents = process_rows(chunk)
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, embeddings)
        else:
            temp_store = FAISS.from_documents(documents, embeddings)
            vector_store.merge_from(temp_store)
    logger.info(f"FAISS vector store created with {vector_store.index.ntotal} documents")
    
    # Save the FAISS index and version file
    try:
        os.makedirs(faiss_index_path, exist_ok=True)
        vector_store.save_local(faiss_index_path)
        logger.info(f"Saved FAISS vector store to {faiss_index_path}")
        with open(version_file, "w") as f:
            f.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save FAISS vector store or version: {str(e)}")
        raise

    return vector_store


def load_dfl_data(google_drive_file_id="1nHdHze3Za5BthXGsk9KptADCLNM7SN0JW4ZI8eIWJCE", faiss_index_path="dfl_faiss_index", version_file="dfl_version.txt"):

    download_url = f"https://docs.google.com/document/d/{google_drive_file_id}/export?format=txt"
    logger.info(f"Downloading Google Doc from {download_url}")
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        text = response.text
        file_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        logger.info(f"Downloaded document and computed hash {file_hash}")
    except Exception as e:
        logger.error(f"Failed to download Google Doc: {str(e)}")
        raise

    if os.path.exists(faiss_index_path) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if stored_hash == file_hash:
                logger.info(
                    f"Precomputed DFL FAISS index found with matching hash at {faiss_index_path}"
                )
                embeddings = get_embeddings()
                return FAISS.load_local(
                    faiss_index_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.info(
                    f"Hash mismatch: stored {stored_hash}, current {file_hash}. Recomputing index."
                )
        except Exception as e:
            logger.error(f"Failed to load precomputed DFL index: {str(e)}. Recomputing.")

    logger.info("Creating new DFL FAISS index")
    chunk_size = 400
    documents = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            documents.append(Document(page_content=chunk, metadata={"chunk": i // chunk_size}))

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    try:
        os.makedirs(faiss_index_path, exist_ok=True)
        vector_store.save_local(faiss_index_path)
        with open(version_file, "w") as vf:
            vf.write(file_hash)
        logger.info(f"Saved DFL FAISS index to {faiss_index_path} and hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save DFL FAISS index: {str(e)}")
        raise

    return vector_store
