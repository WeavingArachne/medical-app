import re
import streamlit as st
from resources import load_resources, DOMAINS
from retrieval import retrieve_codes_batch
from excel_utils import read_excel, update_excel_with_results
import pandas as pd
from io import BytesIO

st.title("Retrieval System")
st.write("Upload your Excel file to find relevant codes.")

# --- Domain selection ---
selected_domain = st.selectbox("Select a domain to search in:", DOMAINS)

if selected_domain:
    index, df_metadata, model = load_resources(selected_domain)
    st.success(
        f"Loaded {selected_domain.capitalize()} resources successfully.")

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        try:
            input_df = pd.read_excel(uploaded_file)
            st.write("Original DataFrame:")
            st.dataframe(input_df.head())

            # --- Step 1: Let user select which columns to use dynamically ---
            all_columns = input_df.columns.tolist()

            st.info("Select the columns you want to use in the retrieval query.")
            selected_columns = st.multiselect(
                "Columns to include in the search query:",
                options=all_columns,
                default=[col for col in all_columns if "SERVICE" in col.upper()]
            )

            if not selected_columns:
                st.warning("Please select at least one column to continue.")
                st.stop()

            st.success(f"Using columns: {', '.join(selected_columns)}")

            # --- Add a Start button ---
            start_process = st.button("🚀 Start Retrieval Process")

            if start_process:
                st.write(
                    "Processing and searching for similar codes... "
                    "This may take a few minutes for large files."
                )

                # --- Step 2: Build the combined query dynamically ---
                def clean_text(text):
                    text = str(text).lower().strip()
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'[^\w\s\-]', '', text)
                    return text

                queries = input_df.apply(
                    lambda row: " | ".join(
                        clean_text(row.get(col, "")) for col in selected_columns
                    ),
                    axis=1
                ).tolist()

                # --- Step 3: Retrieve results in batches ---
                batch_size = 100  # adjust as needed
                results_list = []
                progress_bar = st.progress(0)
                total = len(queries)

                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch_queries = queries[start:end]

                    batch_results = retrieve_codes_batch(
                        model, index, df_metadata, batch_queries, k=4
                    )

                    for retrieved_info in batch_results:
                        flat_info = {}
                        # Loop over each retrieved row (top-k results)
                        for j in range(len(retrieved_info)):
                            for col in retrieved_info.columns:
                                flat_info[f'{col} {j+1}'] = retrieved_info.iloc[j][col]
                        results_list.append(flat_info)

                    progress_bar.progress(end / total)

                results_df = pd.DataFrame(results_list)
                output_df = pd.concat([input_df, results_df], axis=1)

                st.write("Results with retrieved codes:")
                st.dataframe(output_df.head())

                # excel_data = update_excel_with_results(
                #     uploaded_file, results_df)
                uploaded_bytes = BytesIO(uploaded_file.getvalue())
                excel_data = update_excel_with_results(
                    uploaded_bytes, results_df)
                st.download_button(
                    label=f"Download {selected_domain.capitalize()} Results",
                    data=excel_data,
                    file_name=f"codes_{selected_domain}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
