"""Run with python -m streamlit run app.py after installing requirements-llm.txt."""
from pathlib import Path
import streamlit as st
from portfolio.rag import Conversation, Retriever

st.set_page_config(page_title='Document assistant | Manahil Iftikhar', page_icon='📚')
st.title('Document assistant')
st.caption('A small retrieval experiment. Inspect the retrieved passages alongside each answer.')


@st.cache_resource
def load_resources():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    corpus = Path(__file__).parent / 'data' / 'sample-knowledge-base.txt'
    retriever = Retriever.build({corpus.name: corpus.read_text(encoding='utf-8')})
    name = 'google/flan-t5-small'
    return retriever, AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)


if st.button('Start assistant') or 'conversation' in st.session_state:
    with st.spinner('Loading model and document index…'):
        retriever, tokenizer, model = load_resources()
    if 'conversation' not in st.session_state:
        st.session_state.conversation = Conversation(retriever, tokenizer, model)
        st.session_state.messages = []
    if st.button('Clear conversation'):
        st.session_state.conversation.history.clear()
        st.session_state.messages.clear()
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['text'])
            if message.get('sources'):
                with st.expander('Retrieved passages'):
                    for p in message['sources']:
                        st.write(f'{p["source"]} · similarity {p["similarity"]:.2f}')
                        st.write(p['text'])
    if question := st.chat_input('Ask about the sample portfolio documents'):
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            try:
                result = st.session_state.conversation.ask(question)
                st.write(result['answer'])
                with st.expander('Retrieved passages'):
                    for p in result['sources']:
                        st.write(f'{p["source"]} · similarity {p["similarity"]:.2f}')
                        st.write(p['text'])
                st.session_state.messages.extend([
                    {'role': 'user', 'text': question},
                    {'role': 'assistant', 'text': result['answer'], 'sources': result['sources']}])
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
else:
    st.info('Start loads the public model checkpoints. An internet connection is needed on the first run.')
