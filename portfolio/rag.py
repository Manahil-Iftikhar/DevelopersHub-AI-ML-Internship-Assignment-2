"""Small retrieval experiment with explicit sources and session-local history.

The similarity threshold is a heuristic requiring evaluation on the user's corpus.
Retrieved passages are shown as evidence; model-generated answers may be incorrect.
"""
from dataclasses import dataclass, field


def chunk_documents(documents, max_chars=500, overlap=80):
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError('Require max_chars > overlap >= 0.')
    chunks = []
    for source, text in documents.items():
        if not isinstance(text, str):
            raise ValueError('Document values must be strings.')
        for start in range(0, len(text), max_chars-overlap):
            value = text[start:start+max_chars].strip()
            if value:
                chunks.append({'source': source, 'text': value})
            if start + max_chars >= len(text):
                break
    if not chunks:
        raise ValueError('At least one non-empty document is required.')
    return chunks


@dataclass
class Retriever:
    chunks: list
    encoder: object
    index: object

    @classmethod
    def build(cls, documents, encoder=None):
        import faiss
        import numpy as np
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        chunks = chunk_documents(documents)
        vectors = np.asarray(encoder.encode([c['text'] for c in chunks],
                                            normalize_embeddings=True), dtype='float32')
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(chunks, encoder, index)

    def search(self, question, k=3, minimum_similarity=0.35):
        import numpy as np
        if not question.strip() or k < 1:
            raise ValueError('Provide a question and a positive result count.')
        vector = np.asarray(self.encoder.encode([question], normalize_embeddings=True), dtype='float32')
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [{**self.chunks[i], 'similarity': float(score)}
                for i, score in zip(indices[0], scores[0])
                if i >= 0 and score >= minimum_similarity]


@dataclass
class Conversation:
    retriever: object
    tokenizer: object
    model: object
    history: list = field(default_factory=list)
    max_turns: int = 3

    def ask(self, question):
        question = question.strip()
        if not question:
            raise ValueError('Question cannot be empty.')
        # Previous user turns help retrieval resolve short follow-up questions.
        recent = self.history[-self.max_turns:]
        retrieval_query = ' '.join([q for q, _ in recent[-1:]] + [question])
        passages = self.retriever.search(retrieval_query)
        if not passages:
            return {'answer': 'I could not find enough relevant information in these documents.',
                    'sources': [], 'status': 'insufficient_context'}
        context = '\n'.join(f'[{i+1}] {p["text"]}' for i, p in enumerate(passages))
        history = '\n'.join(f'Q: {q}\nA: {a}' for q, a in recent)
        prompt = (f'Answer using only the context. Say when the answer is missing.\n'
                  f'Question: {question}\nContext:\n{context}\nRecent conversation:\n{history}\nAnswer:')
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        import torch
        self.model.eval()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.history.append((question, answer))
        self.history[:] = self.history[-self.max_turns:]
        return {'answer': answer, 'sources': passages, 'status': 'generated_unverified'}
