import numpy as np


def retrieve_codes_batch(model, index, df_metadata, queries, k=4, return_columns=None):
    """
    Retrieve the top-k codes for a batch of queries, dynamically returning
    available metadata columns.

    Parameters
    ----------
    model : SentenceTransformer or similar
        The embedding model used for encoding queries.
    index : FAISS index
        The FAISS index for similarity search.
    df_metadata : pd.DataFrame
        The metadata DataFrame used to map retrieved indices to code details.
    queries : list of str
        List of search queries.
    k : int, optional
        Number of results to retrieve per query. Default is 4.
    return_columns : list of str, optional
        List of columns to return from df_metadata.
        If None, will auto-detect all useful columns (except embeddings).

    Returns
    -------
    results : list of pd.DataFrame
        A list where each element is a DataFrame containing the retrieved
        metadata rows for one query.
    """
    # --- Encode queries ---
    embeddings = model.encode(queries, convert_to_numpy=True).astype('float32')

    # --- Search FAISS index ---
    distances, indices = index.search(embeddings, k)

    # --- Determine which columns to return dynamically ---
    if return_columns is None:
        # Auto-detect columns to include
        # Exclude embedding or technical columns if present
        exclude_cols = {"embedding", "vector", "index", "combined_description"}
        return_columns = [
            c for c in df_metadata.columns if c not in exclude_cols]

    # --- Retrieve metadata for each query ---
    results = []
    for idx_list in indices:
        retrieved = df_metadata.iloc[idx_list].copy()
        results.append(retrieved[return_columns].reset_index(drop=True))

    return results

# import re
# import numpy as np
# import google.generativeai as genai
# import pandas as pd
# import os
# import json
# # Make sure to configure Gemini API key via Streamlit secrets or environment variable
# genai.configure(api_key="AIzaSyAzD2hWCRha-WxJwQCU_w_0IvdsIHonQvM")


# def safe_json_parse(text):
#     """
#     Attempt to safely extract and parse a JSON array (like [1,2,3]) from a text string.
#     Returns None if not valid.
#     """
#     if not text:
#         return None
#     match = re.search(r'\[[^\]]*\]', text, re.DOTALL)
#     if not match:
#         return None
#     try:
#         return json.loads(match.group(0))
#     except json.JSONDecodeError:
#         return None


# def retrieve_codes_batch(model, index, df_metadata, queries, k=4, return_columns=None):
#     """
#     Retrieve top-k codes for a batch of queries.
#     First gets top 10 from FAISS, then re-ranks with Gemini reasoning.
#     """

#     embeddings = model.encode(queries, convert_to_numpy=True).astype('float32')

#     distances, indices = index.search(embeddings, 10)

#     if return_columns is None:
#         exclude_cols = {"embedding", "vector", "index"}
#         return_columns = [
#             c for c in df_metadata.columns if c not in exclude_cols]

#     model_gemini = genai.GenerativeModel("gemini-2.5-flash")
#     results = []

#     for query, idx_list in zip(queries, indices):
#         retrieved = df_metadata.iloc[idx_list][return_columns].copy()
#         retrieved.reset_index(drop=True, inplace=True)

#         gemini_prompt = f"""
# You are ranking medical codes by relevance.

# Query:
# {query}

# Candidate results (indexed 0–9):
# {retrieved.to_dict(orient='records')}

# Return the indices (0-based) of the top {k} most relevant results
# as a **valid JSON array** (e.g. [0,3,5,2]).
# Do not include any explanations or text before or after the JSON.
# Output only raw JSON.
# """

#         try:
#             # First attempt
#             response = model_gemini.generate_content(gemini_prompt)
#             text = (response.text or "").strip()
#             selected_indices = safe_json_parse(text)

#             # Retry if invalid
#             if not selected_indices:
#                 retry_prompt = gemini_prompt + \
#                     "\n\nRemember: Output only a JSON array like [0,1,2,3]."
#                 response = model_gemini.generate_content(retry_prompt)
#                 text = (response.text or "").strip()
#                 selected_indices = safe_json_parse(text)

#             # Final fallback to FAISS if still invalid
#             if not selected_indices:
#                 print(f"[Gemini warning] Invalid JSON output: {repr(text)}")
#                 selected_indices = list(range(k))

#             # Ensure indices are valid
#             selected_indices = [
#                 i for i in selected_indices if 0 <= i < len(retrieved)]
#             top_selected = retrieved.iloc[selected_indices].reset_index(
#                 drop=True)

#         except Exception as e:
#             print(f"[Gemini reasoning error] {e}")
#             print(f"[Gemini raw response] {repr(locals().get('text', ''))}")
#             top_selected = retrieved.head(k)

#         results.append(top_selected)

#     return results
