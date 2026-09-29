"""Exercise the Streamlit app without a browser or any model/API calls."""
from pathlib import Path
from unittest.mock import patch
import socket
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# AppTest is in-process; fail if app code attempts an outbound connection.
with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network access forbidden')):
    app = AppTest.from_file(str(ROOT / 'offline_app.py')).run(timeout=30)
    assert not app.exception, app.exception
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert any('0.7186' in element.value for element in app.text)
    assert any('churn · similarity' in element.value for element in app.markdown)

    app.text_input[0].set_value('zzzxqv').run()
    app.button[0].click().run()
    assert any('No passages met' in element.value for element in app.warning)

    app.text_input[0].set_value(' ').run()
    app.button[0].click().run()
    assert any('Enter a question' in element.value for element in app.warning)

    app.text_input[0].set_value('Is its data real?')
    app.text_input[1].set_value('Tell me about the customer churn experiment')
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert any('churn · similarity' in element.value for element in app.markdown)
    assert any('Tell me about the customer churn experiment Is its data real?' in element.value
               for element in app.caption)

    fresh = AppTest.from_file(str(ROOT / 'offline_app.py')).run(timeout=30)
    assert fresh.text_input[1].value == ''
    assert not fresh.exception, fresh.exception

assert not any(name in sys.modules for name in ('torch', 'transformers', 'sentence_transformers', 'faiss'))
print('PASS: relevant query, no-match, blank input, follow-up context and fresh-session state; no network or model imports')
