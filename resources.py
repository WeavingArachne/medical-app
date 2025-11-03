# resources.py
import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import os


# define available domains
DOMAINS = [
    "Radiograph NICIP",
    "Diagnosis ICD-10",
    "GMDN Medical Devices",
    "Laboratory LOINC",
    "Medication MRID",
    "Procedure ACHI",
    "Dental inpatient",
    "Dental outpatient",
    "SNOMED CT For Allergy and Vaccination",
]


@st.cache_resource(show_spinner=False)
def load_resources(domain: str):
    """
    Load the FAISS index, metadata, and model for a specific domain.
    Cached by domain for efficiency.
    """
    base_path = os.path.join("vectorstores", domain)
    index_path = os.path.join(base_path, "faiss_index.bin")
    metadata_path = os.path.join(base_path, "faiss_metadata.parquet")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing FAISS files for domain '{domain}' in {base_path}")

    st.info(f"Loading resources for **{domain.capitalize()}** domain...")

    index = faiss.read_index(index_path)
    df_metadata = pd.read_parquet(metadata_path)
    model = SentenceTransformer('BAAI/bge-m3')
    # model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

    return index, df_metadata, model
