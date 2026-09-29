"""Local retrieval explorer: streamlit run offline_app.py. No model downloads."""
import json
from pathlib import Path
import streamlit as st
from portfolio.retrieval_eval import evaluate

st.set_page_config(page_title='Retrieval explorer | Manahil Iftikhar', page_icon='📚')
st.title('Document retrieval explorer')
st.caption('Explore six portfolio documents without model downloads or API keys.')
st.info('This demo retrieves passages; it does not generate answers. A related passage may not contain the answer you need.')

corpus_path = Path(__file__).parent / 'evaluations' / 'retrieval.json'
documents = json.loads(corpus_path.read_text(encoding='utf-8'))['documents']

with st.expander('Read the source documents'):
    for source, content in documents.items():
        st.subheader(source)
        st.text(content)

with st.form('retrieval'):
    question = st.text_input('Your question', value='What test ROC-AUC did the churn random forest achieve?')
    previous = st.text_input('Previous question (optional, for a follow-up)', value='')
    submitted = st.form_submit_button('Find passages')

if submitted:
    if not question.strip():
        st.warning('Enter a question to search the documents.')
    else:
        query_case = {'question': question.strip(), 'relevant_sources': []}
        if previous.strip():
            query_case['previous_question'] = previous.strip()
        # Reuse the measured baseline, including chunking, vocabulary and cutoff.
        result = evaluate({'documents': documents, 'cases': [query_case]})['outputs'][0]
        st.caption('Search query: ' + result['retrieval_query'])
        if not result['retrieved']:
            st.warning('No passages met the similarity cutoff. Try another question; this does not prove the answer is absent.')
        else:
            st.subheader('Retrieved passages — inspect before drawing conclusions')
            for index, passage in enumerate(result['retrieved'], 1):
                st.write(f"{index}. {passage['source']} · similarity {passage['similarity']:.3f}")
                st.text(passage['text'])

st.caption('TF-IDF lexical search · top 3 chunks · similarity cutoff 0.15. Similarity is not confidence. The published diagnostic retrieved misleading context for 3 of 4 unanswerable questions.')
