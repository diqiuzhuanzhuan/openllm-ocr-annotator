# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import streamlit as st
import json
from pathlib import Path
import random


@st.cache_data
def load_docs(
    sample_count_for_cache, data_dir_path
):  # Add a parameter to invalidate cache and accept data_dir_path
    docs = []
    for json_file in data_dir_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            docs.append(
                {
                    "id": json_file.stem,
                    "fields": data.get("result", {}).get("fields", []),
                    "image_path": data.get("metadata").get("image_path", ""),
                }
            )
    return docs


def main():
    st.title("Annotation Results Browser")

    # Add DATA_DIR input to sidebar
    default_data_dir = "data/outputs/foreign_trade_20250519/voted_results"
    data_dir_input = st.sidebar.text_input("Data Directory", value=default_data_dir)
    DATA_DIR_PATH = Path(data_dir_input)

    sample_count = st.sidebar.slider(
        "Select number of samples to validate",
        min_value=1,
        max_value=1024,
        value=100,
        step=1,
    )

    docs = load_docs(
        sample_count, DATA_DIR_PATH
    )  # Pass the slider value and DATA_DIR_PATH to the cached function
    doc_ids = [doc["id"] for doc in docs]

    # initialize session_state index if not present
    if "doc_index" not in st.session_state:
        st.session_state.doc_index = 0

    # Store sample_doc_ids in session_state to prevent regeneration on every rerun
    if (
        "sample_doc_ids" not in st.session_state
        or st.session_state.get("sample_count_for_session") != sample_count
    ):
        st.session_state.sample_doc_ids = random.sample(
            doc_ids, min(sample_count, len(doc_ids))
        )
        st.session_state.sample_count_for_session = sample_count
        st.session_state.doc_index = 0  # Reset index when sample changes
        st.session_state.all_checked_fields = {}  # Reset all_checked_fields when sample changes

    sample_doc_ids = st.session_state.sample_doc_ids

    selected_doc_id = sample_doc_ids[st.session_state.doc_index]
    selected_doc = next(doc for doc in docs if doc["id"] == selected_doc_id)

    st.header(
        f"Document: {selected_doc_id} ({st.session_state.doc_index + 1}/{len(sample_doc_ids)})"
    )

    if selected_doc["image_path"]:
        st.image(
            selected_doc["image_path"],
            caption="Document Image",
            use_container_width=True,
        )
    else:
        st.warning("No image path found.")

    st.subheader("Fields")

    # Initialize session_state for all checked fields across documents
    if "all_checked_fields" not in st.session_state:
        st.session_state.all_checked_fields = {}

    # Initialize checked_fields for the current document if not present
    if selected_doc_id not in st.session_state.all_checked_fields:
        st.session_state.all_checked_fields[selected_doc_id] = {
            field.get("field_name", ""): True for field in selected_doc["fields"]
        }

    current_doc_checked_fields = st.session_state.all_checked_fields[selected_doc_id]

    for field in selected_doc["fields"]:
        confidence = field.get("confidence", 1.0)
        field_name = field.get("field_name", "")
        value = field.get("value", "")
        label = f"{field_name}: {value} (Confidence: {confidence:.2f})"
        if confidence < 0.85:
            label += " ⚠️ Low confidence"

        # Use a unique key for each checkbox and update the session state directly
        checked = st.checkbox(
            label,
            value=current_doc_checked_fields.get(field_name, True),
            key=f"{selected_doc_id}_{field_name}",
        )
        current_doc_checked_fields[field_name] = checked

    # Calculate and display overall accuracy across all sampled documents
    total_fields_overall = 0
    correct_fields_overall = 0

    for doc_id, fields_state in st.session_state.all_checked_fields.items():
        total_fields_overall += len(fields_state)
        correct_fields_overall += sum(1 for v in fields_state.values() if v)

    if total_fields_overall > 0:
        overall_accuracy = (correct_fields_overall / total_fields_overall) * 100
        st.metric(
            label="Overall Accuracy Across Sampled Documents",
            value=f"{overall_accuracy:.2f}%",
        )
    else:
        st.info("No fields to evaluate across sampled documents.")

    # Remove the Submit Checked Fields button as accuracy is dynamic
    # if st.button("Submit Checked Fields"):
    #     st.write("Checked fields:", {k: v for k, v in checked_fields.items() if v})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            if st.session_state.doc_index > 0:
                st.session_state.doc_index -= 1
                st.rerun()
    with col2:
        if st.button("Next"):
            if st.session_state.doc_index < len(sample_doc_ids) - 1:
                st.session_state.doc_index += 1
                st.toast("next document loaded")
                st.rerun()


if __name__ == "__main__":
    main()
