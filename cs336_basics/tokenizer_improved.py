"""
改进的BPE Tokenizer实现
结合参考实现的优点和我们的功能
"""

import re
import json
import ast
from typing import Iterable, Iterator, Optional
from pathlib import Path

# GPT-2预分词正则表达式（需要regex模块支持\p{}语法）
try:
    import regex as regex_mod
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def find_tokens(text):
        pat = regex_mod.compile(PAT)
        return [m.group(0) for m in pat.finditer(text)]
except ImportError:
    # 回退到标准re模块的简化版本
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    def find_tokens(text):
        return [m.group(0) for m in re.finditer(PAT, text)]

def to_bytes_tuple(word: str) -> tuple[bytes, ...]:
    """将字符串转换为单个字节的元组"""
    return tuple(bytes([b]) for b in word.encode("utf-8"))


class Tokenizer:
    """
    改进的BPE Tokenizer类
    结合参考实现的优点和我们的功能
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        构建BPE tokenizer
        
        Args:
            vocab: token ID到字节表示的映射
            merges: BPE合并操作列表
            special_tokens: 应被视为不可分割token的字符串列表
        """
        self.vocab = vocab
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        
        # 构建BPE优先级映射（参考实现的方法）
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # 处理特殊token（参考实现的改进方法）
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        # 确保特殊token在词汇表中
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        """
        从文件加载tokenizer（保留我们的功能）
        """
        # 加载词汇表
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        vocab: dict[int, bytes] = {int(k): bytes(v) for k, v in raw_vocab.items()}

        # 加载合并规则
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 处理复杂格式
                if line.startswith("b'") and " b'" in line:
                    second_b_pos = line.find(" b'")
                    if second_b_pos != -1:
                        left_part = line[:second_b_pos]
                        right_part = line[second_b_pos + 1:]
                    else:
                        continue
                else:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    left_part = parts[0]
                    right_part = parts[1]
                
                try:
                    left = ast.literal_eval(left_part)
                    right = ast.literal_eval(right_part)
                    if not isinstance(left, (bytes, bytearray)) or not isinstance(right, (bytes, bytearray)):
                        continue
                    merges.append((bytes(left), bytes(right)))
                except:
                    continue

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        编码文本为token ID序列（参考实现的改进方法）
        """
        tokens = []

        # 按长度排序特殊token（最长优先）避免部分匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                # 如果是特殊token，直接添加其ID
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                # 否则使用BPE正常tokenize
                tokens.extend(self._tokenize_normal(part))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对可迭代字符串进行内存高效的编码
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """
        解码token ID序列为人类可读字符串（参考实现的简洁方法）
        """
        # 连接所有token字节
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        # 解码字节为字符串，替换无效序列
        return full_bytes.decode("utf-8", errors="replace")

    def _tokenize_normal(self, text: str) -> list[int]:
        """
        对普通文本进行tokenize（参考实现的方法）
        """
        # 预分词
        pre_tokens = find_tokens(text)

        token_ids = []
        for token in pre_tokens:
            # 转换token为字节元组
            byte_tuple = to_bytes_tuple(token)
            
            # 应用BPE合并（对每个预分词token分别处理）
            merged = self._apply_merges(byte_tuple)
            
            # 获取token IDs
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        
        return token_ids

    def _apply_merges(self, byte_tuple: tuple[bytes, ...]) -> list[bytes]:
        """
        对字节序列应用BPE合并（参考实现的高效方法）
        """


        word: list[bytes] = list(byte_tuple)

        def get_pairs(word: list[bytes]):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            # 找到最高优先级的合并（参考实现的方法）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))

            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        return word


# 测试函数
def test_improved_tokenizer():
    """测试改进的tokenizer"""
    # 创建简单的测试vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # 添加一些合并的token
    vocab[256] = b'th'
    vocab[257] = b'he'
    vocab[258] = b'lo'
    vocab[259] = b'hello'
    
    merges = [(b't', b'h'), (b'h', b'e'), (b'l', b'o'), (b'he', b'llo')]
    
    tokenizer = Tokenizer(vocab, merges, special_tokens=['<|endoftext|>'])
    
    # 测试基本功能
    text = "hello"
    result = tokenizer.encode(text)
    print(f"输入: '{text}'")
    print(f"编码结果: {result}")
    print(f"解码结果: '{tokenizer.decode(result)}'")
    
    # 测试特殊token
    text_with_special = "hello <|endoftext|> world"
    result2 = tokenizer.encode(text_with_special)
    print(f"\n输入: '{text_with_special}'")
    print(f"编码结果: {result2}")
    print(f"解码结果: '{tokenizer.decode(result2)}'")


if __name__ == "__main__":
    test_improved_tokenizer()
