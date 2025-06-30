#########
# 7. app.py
#########

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/search"

st.title("📄 Research Paper Recommender")
st.markdown("Type a research paper title or abstract to get similar papers.")

# Initialize session state for error handling
if 'api_error' not in st.session_state:
    st.session_state.api_error = False

query = st.text_area("🔍 Your Query", height=150)
k = st.slider("Top K Results", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        try:
            # Add timeout and retry logic
            response = requests.get(
                API_URL,
                params={"query": query, "k": k},
                timeout=10
            )
            response.raise_for_status()
            results = response.json()
            
            st.session_state.api_error = False
            
            st.subheader("🔎 Top Matches:")
            for r in results:
                st.markdown(f"**{r['title']}**  \n`Score: {r['score']:.3f}`  \nPaper ID: `{r['id']}`")
                st.markdown("---")
                
        except requests.exceptions.RequestException as e:
            st.session_state.api_error = True
            st.error(f"⚠️ API Connection Failed: {str(e)}")
            st.markdown("""
            **Troubleshooting Steps:**
            1. Ensure the FastAPI server is running (`uvicorn api.main:app --reload`)
            2. Check the API URL is correct (currently: `{API_URL}`)
            3. Verify no firewall is blocking port 8000
            """)

# Show persistent error message if API is down
if st.session_state.api_error:
    st.warning("The recommendation service is currently unavailable. Please try again later.")