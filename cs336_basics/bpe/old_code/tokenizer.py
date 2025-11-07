"""
BPE Tokenizer Implementation
按照 CS336 Assignment 1 的要求实现 Tokenizer 类
"""

import torch
from typing import Iterable, Iterator, Optional
import os
import ast
import json
from pathlib import Path


class Tokenizer:
    """
    BPE Tokenizer 类
    
    功能：
    1. 从词汇表和合并规则构建 tokenizer
    2. 将文本编码为 token IDs
    3. 将 token IDs 解码为文本
    4. 支持特殊 token 处理
    5. 支持内存高效的流式处理
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None):
        """
        从给定的词汇表、合并规则和特殊 token 构建 tokenizer
        
        Args:
            vocab: 词汇表，映射 token ID 到字节序列
            merges: BPE 合并规则列表
            special_tokens: 特殊 token 列表（可选）
        """
        # 1. 存储基本数据
        self.vocab = vocab.copy()  # 使用 copy() 避免修改原始数据
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 2. 处理特殊 token
        if special_tokens:
            next_id = max(vocab.keys()) + 1
            for special_token in special_tokens:
                special_bytes = special_token.encode('utf-8')
                if special_bytes not in self.vocab.values():
                    self.vocab[next_id] = special_bytes
                    next_id += 1
        
        # 3. 构建反向映射
        self.id_to_bytes = {}
        self.bytes_to_id = {}
        
        for token_id, token_bytes in self.vocab.items():
            self.id_to_bytes[token_id] = token_bytes
            self.bytes_to_id[token_bytes] = token_id
        
        # 4. 构建合并规则映射
        self.merges_map = {}
        for a, b in merges:
            merged_bytes = a + b
            self.merges_map[(a, b)] = merged_bytes

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        """
        从序列化的词汇表和合并规则文件构建 tokenizer
        
        Args:
            vocab_filepath: 词汇表文件路径
            merges_filepath: 合并规则文件路径
            special_tokens: 特殊 token 列表（可选）
            
        Returns:
            Tokenizer 实例
        """
        # 加载词汇表（JSON，键为字符串的ID，值为字节的整数列表）
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        vocab: dict[int, bytes] = {int(k): bytes(v) for k, v in raw_vocab.items()}

        # 加载合并规则（文本文件，每行两个bytes的repr，例如：b'th' b'e'）
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 处理格式如 "b' ' b'p'" 或 "b' p' b'a'" 的情况
                # 需要找到第一个 b' 和第二个 b' 的位置
                if line.startswith("b'") and " b'" in line:
                    # 找到第二个 b' 的位置
                    second_b_pos = line.find(" b'")
                    if second_b_pos != -1:
                        left_part = line[:second_b_pos]
                        right_part = line[second_b_pos + 1:]  # 去掉开头的空格
                    else:
                        continue
                else:
                    # 处理普通格式 "b'th' b'e'"
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    left_part = parts[0]
                    right_part = parts[1]
                
                # 使用 ast.literal_eval 将 b'..' 的repr 转回 bytes
                try:
                    left = ast.literal_eval(left_part)
                    right = ast.literal_eval(right_part)
                    if not isinstance(left, (bytes, bytearray)) or not isinstance(right, (bytes, bytearray)):
                        continue
                    merges.append((bytes(left), bytes(right)))
                except:
                    continue

        return cls(vocab, merges, special_tokens)

    def _to_bytes_tuple(self, word: str) -> tuple[bytes, ...]:
        """
        将字符串转换为单个字节的元组
        
        Args:
            word: 输入字符串
            
        Returns:
            单个字节的元组
        """
        l = list(word.encode("utf-8"))
        l = [bytes([x]) for x in l]
        return tuple(l)
    
    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token ID 序列
        
        Args:
            text: 输入文本
            
        Returns:
            token ID 列表
        """
        # 首先处理 special tokens
        result_ids = []
        remaining_text = text
        
        while remaining_text:
            # 查找最早出现的 special token（优先匹配更长的）
            earliest_special = None
            earliest_pos = len(remaining_text)
            
            # 按长度降序排序 special tokens，优先匹配更长的
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            
            for special_token in sorted_special_tokens:
                pos = remaining_text.find(special_token)
                if pos != -1 and pos < earliest_pos:
                    earliest_special = special_token
                    earliest_pos = pos
            
            if earliest_special is not None:
                # 处理 special token 之前的文本
                if earliest_pos > 0:
                    before_text = remaining_text[:earliest_pos]
                    before_ids = self._encode_text(before_text)
                    result_ids.extend(before_ids)
                
                # 添加 special token
                special_id = self.bytes_to_id.get(earliest_special.encode('utf-8'))
                if special_id is not None:
                    result_ids.append(special_id)
                
                # 更新剩余文本
                remaining_text = remaining_text[earliest_pos + len(earliest_special):]
            else:
                # 没有更多 special tokens，处理剩余文本
                remaining_ids = self._encode_text(remaining_text)
                result_ids.extend(remaining_ids)
                break
        
        return result_ids
    
    def _encode_text(self, text: str) -> list[int]:
        """
        编码普通文本（不包含 special tokens）
        """
        pretokens = self._pretokenize(text)  # list[str]

        
        # 对每个预分词后的token分别应用BPE合并!!!而不是
        all_merged_tokens = []
        for token in pretokens:
            # 将单个token转换为字节元组
            byte_tuple = self._to_bytes_tuple(token)
            # 对每个token分别应用合并
            merged_token = self._apply_merges(list(byte_tuple))
            all_merged_tokens.extend(merged_token)
        
        ids = self._bytes_to_ids(all_merged_tokens)  # list[int]
        
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对可迭代的字符串进行内存高效的编码
        
        Args:
            iterable: 字符串可迭代对象（如文件句柄）
            
        Yields:
            token ID（惰性生成）
        """
        # 实现流式编码
        # 1. 遍历输入的可迭代对象
        # 2. 对每个字符串调用 encode 方法
        # 3. 惰性地产生 token IDs
        
        for text in iterable:
            # 对每个字符串进行编码
            tokens = self.encode(text)
            # 逐个产生 token ID
            for token_id in tokens:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 序列解码为文本
        
        Args:
            ids: token ID 列表
            
        Returns:
            解码后的文本
        """
        # 实现解码逻辑
        # 1. 将每个 token ID 映射回字节序列
        # 2. 连接所有字节序列
        # 3. 解码为 Unicode 字符串
        # 4. 处理解码错误（使用 U+FFFD 替换字符）
        
        bytes_tokens = self._ids_to_bytes(ids)

        concat_bytes = b''.join(bytes_tokens)

        text = concat_bytes.decode('utf-8', errors='replace')
        return text


    
    def _pretokenize(self, text: str) -> list[str]:
        """
        使用 GPT-2 正则表达式进行预分词
        
        Args:
            text: 输入文本
            
        Returns:
            预分词后的 token 列表
        """
        # 实现预分词
        # 使用 GPT-2 的正则表达式模式
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        import regex as regex_mod
        pat = regex_mod.compile(PAT)
        return [m.group(0) for m in pat.finditer(text)]


    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """
        对字节 token 应用 BPE 合并规则
        
        Args:
            tokens: 字节 token 列表
            
        Returns:
            合并后的字节 token 列表
        """
        # 实现 BPE 合并（按训练时的创建顺序应用 merges）
        # 1. 遍历合并规则（顺序很重要）
        # 2. 对每条规则，在当前序列中左到右扫描，遇到匹配的相邻对则合并
        # 3. 直到所有规则应用完毕，返回结果

        # 取训练得到的 merges 列表，从第一条到最后一条逐条应用；
        # 对每一条 (a, b)，在序列中“左到右”扫描，遇到相邻的 a, b 就合并为 a+b，然后继续扫描直到该规则在该序列上不再匹配；
        # 完成这一条规则后，继续下一条规则；
        # 规则之间的顺序必须保持和训练时一致。

        if not tokens or len(tokens) < 2:
            return tokens

        out = tokens[:]

        # 遍历目前所有的已经存在的merges
        for a, b in self.merges:
            merged = self.merges_map.get((a, b))
            if merged is None:
                continue
            i = 0
            # 在当前序列上反复扫描，左到右合并本规则的匹配对
            while i < len(out) - 1:
                # 匹配到了merge pair
                if out[i] is a or out[i] == a:
                    if out[i + 1] is b or out[i + 1] == b:
                        # 合并 i 和 i+1
                        out[i] = merged
                        del out[i + 1]
                        # 合并后，继续在当前位置尝试下一对（可能与前一个产生新匹配）
                        if i > 0:
                            i -= 1
                        continue
                i += 1
        return out
    
    def _bytes_to_ids(self, tokens: list[bytes]) -> list[int]:
        """
        将字节 token 转换为 token IDs
        
        Args:
            tokens: 字节 token 列表
            
        Returns:
            token ID 列表
        """
        # 实现字节到 ID 的转换
        # 使用反向词汇表映射
        
        ids = []
        for token in tokens:
            if token in self.bytes_to_id:
                ids.append(self.bytes_to_id[token])
            else:
                raise ValueError(f"Token {token!r} not found in vocab.")
        return ids

    
    def _ids_to_bytes(self, ids: list[int]) -> list[bytes]:
        """
        将 token IDs 转换为字节 token
        
        Args:
            ids: token ID 列表
            
        Returns:
            字节 token 列表
        """
        # 实现 ID 到字节的转换
        # 使用词汇表映射
        
        tokens = []
        for id in ids:
            if id in self.id_to_bytes:
                tokens.append(self.id_to_bytes[id])
            else:
                raise ValueError(f"ID {id!r} not found in vocab.")
        
        return tokens





def test_from_files(dataset_prefix: str = "tinystories") -> None:
    """从磁盘加载已保存的 vocab/merges 做一次烟雾测试。

    Args:
        dataset_prefix: 结果文件前缀，例如 "tinystories" 或 "owt"。
    """
    base = Path(__file__).parent / "bpe" / "bpe_results"
    vocab_fp = base / f"{dataset_prefix}_vocab.json"
    merges_fp = base / f"{dataset_prefix}_merges.txt"

    if not vocab_fp.exists() or not merges_fp.exists():
        print(f"bpe_results not found for prefix '{dataset_prefix}'; skipping from_files smoke test.")
        return

    print(f"Loading vocab from: {vocab_fp}")
    print(f"Loading merges from: {merges_fp}")
    tok = Tokenizer.from_files(str(vocab_fp), str(merges_fp), special_tokens=["<|endoftext|>"])

    print(f"Vocab size: {len(tok.vocab):,}")
    print(f"Merges count: {len(tok.merges_map):,}")

    learned = [(tid, tb) for tid, tb in sorted(tok.vocab.items()) if tid >= 256][:3]
    for tid, tb in learned:
        try:
            s = tb.decode('utf-8', errors='replace')
        except Exception:
            s = "<decode-error>"
        print(f"Sample learned token: id={tid} bytes={tb!r} str='{s}'")

    # 基础类型断言
    assert isinstance(next(iter(tok.vocab.keys())), int)
    assert isinstance(next(iter(tok.vocab.values())), (bytes, bytearray))
    mk = next(iter(tok.merges_map.keys()))
    assert isinstance(mk, tuple) and len(mk) == 2 and isinstance(mk[0], (bytes, bytearray))
    print("from_files load OK.")


def test_pretokenize_basic() -> None:
    """基于 handout 示例的预分词单元测试。"""
    tok = Tokenizer(vocab={}, merges=[], special_tokens=None)
    text = "some text that i'll pre-tokenize"
    out = tok._pretokenize(text)
    expected = ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
    print("pretok ->", out)
    assert out == expected, f"_pretokenize output mismatch. got={out} expected={expected}"
    print("_pretokenize basic test OK.")


def test_encode_basic() -> None:
    """测试 encode 方法的基本功能"""
    # 创建一个简单的 tokenizer 实例用于测试
    # 包含基本字节词汇表 (0-255)
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # 添加一些合并的token用于测试
    vocab[256] = b'th'
    vocab[257] = b'he'
    vocab[258] = b'lo'
    vocab[259] = b'hello'  # 添加完整的单词
    vocab[260] = b'world'  # 添加完整的单词
    vocab[261] = b' world'  # 添加带空格的单词
    merges = [(b't', b'h'), (b'h', b'e'), (b'l', b'o'), (b'he', b'llo')]
    
    tokenizer = Tokenizer(vocab, merges)
    
    # 测试基本编码
    text = "hello"
    result = tokenizer.encode(text)
    print(f"输入: '{text}'")
    print(f"编码结果: {result}")
    
    # 测试包含空格的文本
    text2 = "hello world"
    result2 = tokenizer.encode(text2)
    print(f"\n输入: '{text2}'")
    print(f"编码结果: {result2}")
    
    # 测试空字符串
    text3 = ""
    result3 = tokenizer.encode(text3)
    print(f"\n输入: '{text3}'")
    print(f"编码结果: {result3}")
    
    # 测试单个字符
    text4 = "a"
    result4 = tokenizer.encode(text4)
    print(f"\n输入: '{text4}'")
    print(f"编码结果: {result4}")
    
    print("\n=== 测试 decode 方法 ===")
    
    # 测试 decode 基本功能
    ids1 = [259]  # "hello"
    decoded1 = tokenizer.decode(ids1)
    print(f"Token IDs: {ids1}")
    print(f"解码结果: '{decoded1}'")
    
    # 测试 decode 多个token
    ids2 = [259, 261]  # "hello world"
    decoded2 = tokenizer.decode(ids2)
    print(f"\nToken IDs: {ids2}")
    print(f"解码结果: '{decoded2}'")
    
    # 测试 decode 空列表
    ids3 = []
    decoded3 = tokenizer.decode(ids3)
    print(f"\nToken IDs: {ids3}")
    print(f"解码结果: '{decoded3}'")
    
    # 测试 encode + decode 往返
    original_text = "hello world"
    encoded = tokenizer.encode(original_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n往返测试:")
    print(f"原始文本: '{original_text}'")
    print(f"编码结果: {encoded}")
    print(f"解码结果: '{decoded}'")
    print(f"往返成功: {original_text == decoded}")
    
    print("\nencode/decode basic test completed.")


def test_encode_iterable() -> None:
    """测试 encode_iterable 方法的基本功能"""
    # 创建一个简单的 tokenizer 实例用于测试
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # 添加一些合并的token用于测试
    vocab[256] = b'th'
    vocab[257] = b'he'
    vocab[258] = b'lo'
    vocab[259] = b'hello'
    vocab[260] = b'world'
    vocab[261] = b' world'
    vocab[262] = b'how'
    vocab[263] = b' are'
    vocab[264] = b' you'
    merges = [(b't', b'h'), (b'h', b'e'), (b'l', b'o'), (b'he', b'llo')]
    
    tokenizer = Tokenizer(vocab, merges)
    
    # 测试字符串列表
    texts = ["hello", " world", "how", " are", " you"]
    print("测试字符串列表:")
    print(f"输入: {texts}")
    
    # 使用 encode_iterable
    tokens = list(tokenizer.encode_iterable(texts))
    print(f"encode_iterable 结果: {tokens}")
    
    # 对比使用普通 encode
    normal_tokens = []
    for text in texts:
        normal_tokens.extend(tokenizer.encode(text))
    print(f"普通 encode 结果: {normal_tokens}")
    print(f"结果一致: {tokens == normal_tokens}")
    
    # 测试空列表
    empty_tokens = list(tokenizer.encode_iterable([]))
    print(f"\n空列表测试: {empty_tokens}")
    
    # 测试单个字符串
    single_tokens = list(tokenizer.encode_iterable(["hello"]))
    print(f"单个字符串测试: {single_tokens}")
    
    # 测试惰性生成（不转换为列表）
    print(f"\n惰性生成测试:")
    count = 0
    for token in tokenizer.encode_iterable(texts):
        print(f"Token {count}: {token}")
        count += 1
        if count >= 3:  # 只显示前3个
            print("...")
            break
    
    print("encode_iterable test completed.")


def test_tinystories_pausing():
    """测试 TinyStories tokenizer 对 ' pausing' 的编码"""
    print("测试 TinyStories tokenizer 对 ' pausing' 的编码...")
    
    try:
        # 加载 TinyStories tokenizer
        tokenizer = Tokenizer.from_files(
            'cs336_basics/bpe/bpe_results/tinystories_vocab.json',
            'cs336_basics/bpe/bpe_results/tinystories_merges.txt',
            special_tokens=['<|endoftext|>']
        )
        
        print(f"✅ 成功加载 TinyStories tokenizer")
        print(f"   词汇表大小: {len(tokenizer.vocab)}")
        print(f"   合并规则数量: {len(tokenizer.merges)}")
        
        # 测试 ' pausing' 的编码过程
        text = ' eigenvalue'
        print(f"\n测试文本: '{text}'")
        
        # 1. 预分词
        pretokens = tokenizer._pretokenize(text)
        print(f"1. 预分词结果: {pretokens}")
        
        # 2. 字节转换
        byte_tokens = [token.encode('utf-8') for token in pretokens]
        print(f"2. 字节 tokens: {byte_tokens}")
        
        # 3. 应用合并
        merged_tokens = tokenizer._apply_merges(byte_tokens)
        print(f"3. 合并后 tokens: {merged_tokens}")
        
        # 4. 检查每个 token 是否在词汇表中
        print(f"4. 检查词汇表匹配:")
        for i, token in enumerate(merged_tokens):
            if token in tokenizer.bytes_to_id:
                token_id = tokenizer.bytes_to_id[token]
                print(f"   Token {i} '{token}' -> ID {token_id} ✅")
            else:
                print(f"   Token {i} '{token}' -> NOT FOUND ❌")
        
        # 5. 完整编码测试
        try:
            result = tokenizer.encode(text)
            print(f"5. 完整编码结果: {result} ✅")
            
            # 6. 解码测试
            decoded = tokenizer.decode(result)
            print(f"6. 解码结果: '{decoded}' ✅")
            print(f"   原始文本: '{text}'")
            print(f"   解码匹配: {decoded == text} {'✅' if decoded == text else '❌'}")
            
        except Exception as e:
            print(f"5. 完整编码失败: {e} ❌")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    # test_from_files("tinystories")
    test_tinystories_pausing()
    # test_pretokenize_basic()
    # test_tokenizer()
    # test_encode_basic()
