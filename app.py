import streamlit as st
from generator import generate_flashcards
from PyPDF2 import PdfReader
import pandas as pd

st.set_page_config(page_title="LLM Flashcard Generator", layout="centered")
st.title("📚 LLM Flashcard Generator")
st.markdown("Generate study flashcards using a Hugging Face LLM via LangChain.")

# Subject dropdown
subject = st.selectbox("📘 Select Subject (optional)", ["General", "Biology", "History", "Computer Science", "Math", "Physics"])

# Input method
input_method = st.radio("📥 Choose Input Method", ["Upload PDF/TXT", "Paste Text"])

# Input content
content = ""

if input_method == "Upload PDF/TXT":
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            try:
                pdf = PdfReader(uploaded_file)
                content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        elif uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
else:
    content = st.text_area("Paste your educational content here:", height=300)

# Generate button
if st.button("🚀 Generate Flashcards"):
    if not content.strip():
        st.warning("Please upload or paste some content.")
    else:
        with st.spinner("Generating flashcards using the LLM..."):
            flashcards = generate_flashcards(content, subject)

        if flashcards:
            st.success(f"✅ Generated {len(flashcards)} flashcards!")

            # Display as table
            df = pd.DataFrame(flashcards)
            st.dataframe(df, use_container_width=True)

            # Export buttons
            st.download_button("📥 Download as CSV", df.to_csv(index=False).encode(), "flashcards.csv", "text/csv")
            st.download_button("📥 Download as JSON", df.to_json(orient="records", indent=2).encode(), "flashcards.json", "application/json")
        else:
            st.error("No flashcards could be generated. Try again with better content.")
