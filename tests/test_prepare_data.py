"""Retrieval inputs follow each model's chat format and exact token budget."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import transformers

spec = importlib.util.spec_from_file_location('prepare_data', Path(__file__).parents[1] / 'scripts/prepare_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('family', ['qwen2', 'mistral'])
def test_retrieval_uses_native_message_roles_and_local_tokenizer(tmp_path, monkeypatch, family):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            expected = ['user'] if family == 'mistral' else ['system', 'user']
            assert [m['role'] for m in messages] == expected
            return '<s>' + messages[-1]['content'] + '</s>'

        def encode(self, text, **kwargs):
            return list(text.encode())

    def tokenizer_from_pretrained(path, **kwargs):
        assert kwargs['local_files_only'] is True
        return Tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', tokenizer_from_pretrained)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **k: SimpleNamespace(model_type=family))
    target = tmp_path / 'data.jsonl'
    monkeypatch.setattr(sys, 'argv', ['prepare_data', '--dataset', 'passkey', '--model', str(tmp_path),
        '--local-files-only', '--input-tokens', '1024', '--output', str(target)])
    module.main()
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 3
    assert all(len(row['input_ids']) == 1024 for row in rows)
    assert all(row['dataset'] == 'passkey' for row in rows)
    assert len({row['dataset_fingerprint'] for row in rows}) == 1
