#!/usr/bin/env python3

from cs336_basics.tokenizer import Tokenizer
import regex
import tiktoken

# 测试文本
text = '<|endoftext|>\n\ntesting!'
print(f'原始文本: {repr(text)}')

# 测试预分词
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pat = regex.compile(PAT)
pretokens = [m.group(0) for m in pat.finditer(text)]
print(f'预分词结果: {pretokens}')

# 测试tiktoken的预分词
ref_tokenizer = tiktoken.get_encoding('gpt2')
ref_ids = ref_tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print(f'tiktoken编码: {ref_ids}')

# 查看每个token对应的文本
for i, token_id in enumerate(ref_ids):
    token_bytes = ref_tokenizer.decode([token_id])
    print(f'Token {i}: {token_id} -> {repr(token_bytes)}')

# 测试我们的tokenizer
tokenizer = Tokenizer.from_files('tests/fixtures/vocab.json', 'tests/fixtures/merges.txt', ['<|endoftext|>'])
our_ids = tokenizer.encode(text)
print(f'我们的编码: {our_ids}')

# 查看我们的每个token
for i, token_id in enumerate(our_ids):
    if token_id in tokenizer.id_to_bytes:
        token_bytes = tokenizer.id_to_bytes[token_id]
        print(f'我们的Token {i}: {token_id} -> {repr(token_bytes)}')
    else:
        print(f'我们的Token {i}: {token_id} -> (未找到)')
