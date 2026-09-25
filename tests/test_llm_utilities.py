import unittest
from portfolio.tickets import build_prompt, validate_tags
from portfolio.rag import chunk_documents, Conversation


class TicketTests(unittest.TestCase):
    def test_unrecognized_tags_require_review(self):
        result = validate_tags('Billing, Free Refund, billing')
        self.assertEqual(result['tags'], ['Billing'])
        self.assertEqual(result['unrecognized'], ['Free Refund'])
        self.assertTrue(result['needs_review'])

    def test_empty_output_requires_review(self):
        self.assertTrue(validate_tags('')['needs_review'])

    def test_only_exact_known_labels_are_accepted(self):
        result = validate_tags('login problem; Account Management')
        self.assertEqual(result['tags'], ['Login Problem', 'Account Management'])
        self.assertFalse(result['needs_review'])

    def test_few_shot_examples_are_used(self):
        self.assertIn('I forgot my password', build_prompt('test', few_shot=True))
        self.assertNotIn('I forgot my password', build_prompt('test'))

    def test_blank_input_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt('   ')


class RetrievalTests(unittest.TestCase):
    def test_chunks_keep_source_and_overlap(self):
        result = chunk_documents({'note.txt': 'abcdefghij'}, max_chars=6, overlap=2)
        self.assertEqual(result, [{'source': 'note.txt', 'text': 'abcdef'},
                                  {'source': 'note.txt', 'text': 'efghij'}])

    def test_invalid_chunk_settings_rejected(self):
        with self.assertRaises(ValueError):
            chunk_documents({'a': 'text'}, max_chars=4, overlap=4)

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            chunk_documents({'a': '   '})

    def test_histories_are_per_conversation(self):
        a, b = Conversation(None, None, None), Conversation(None, None, None)
        a.history.append(('q', 'a'))
        self.assertEqual(b.history, [])

    def test_no_evidence_does_not_call_generator(self):
        class EmptyRetriever:
            def search(self, query):
                return []
        result = Conversation(EmptyRetriever(), None, None).ask('unknown topic')
        self.assertEqual(result['status'], 'insufficient_context')
        self.assertEqual(result['sources'], [])


if __name__ == '__main__':
    unittest.main()
