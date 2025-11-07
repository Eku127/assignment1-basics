#!/usr/bin/env python3
"""
训练BPE分词器在TinyStories数据集上 - 带进度条版本

训练感想：
在训练的过程中会发现 total_pair_counts 会不断增加，这个现象是合理的！
因为在训练的初期, 合并的只有很少的pairs,但是新产生的token会产生大量新的token!
比如这个例子:
[DEBUG] --- MERGE STEP 1 ---
[DEBUG] Merging pair: (b' ', b't') -> ID 257
[DEBUG] Initial len(total_pair_counts): 932
    (-) Pairs REMOVED (2 unique types): { (b' ', b't'), (b't', b'v') }
    (+) Pairs ADDED   (10 unique types): { (b' t', b'a'), (b' t', b'e'), (b' t', b'h'), (b' t', b'i'), (b' t', b'o'), (b' t', b'r'), (b' t', b'u'), (b' t', b'v'), (b' t', b'w'), (b' t', b'y') }
[DEBUG] Final len(total_pair_counts): 940
[DEBUG] Net change in len for this step: 8  (10 added - 2 removed)

可以发现, 合并了 (b' ', b't') 和 (b't', b'v') 之后, 新产生了 10 个新的token! 
之前(b't', b'h')如果存在的话，那么有了新的token b' t'之后，就会还会产生另外一个新的pair(b' t', b'h')!
这也就是为什么会出现一直上涨的情况

这个现象完全取决于语料库的大小以及vocab_size的设定，如果vocab size设置足够大的话，一定能看到pairs的上升和下降

可以使用merge_pair_debug函数来看到这个过程

"""

import time
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
import regex as regex_mod
import re

merge_step_counter = 0


def _pretokenize_chunk_batch(chunks: list[str]) -> dict[tuple[int, ...], int]:
    """批量预分词多个文档块，减少进程间通信"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex_mod.compile(PAT)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    
    for chunk in chunks:
        for m in pat.finditer(chunk):
            token = m.group(0)
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    
    return dict(freq)


def _pretokenize_parallel_optimized(segments: list[str], num_processes: int = None) -> dict[tuple[int, ...], int]:
    """优化的多进程并行预分词"""
    if num_processes is None:
        num_processes = min(cpu_count(), len(segments))
    
    # 将文档分批，减少进程间通信次数
    batch_size = max(1, len(segments) // (num_processes * 4))  # 每个进程处理多个文档
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    
    print(f"   使用 {num_processes} 个进程处理 {len(batches)} 个批次...")
    
    # 使用进程池并行处理批次
    with Pool(processes=num_processes) as pool:
        batch_freqs = list(tqdm(
            pool.imap(_pretokenize_chunk_batch, batches),
            total=len(batches),
            desc="预分词进度",
            unit="批次"
        ))
    
    # 合并所有结果
    total_freq: dict[tuple[int, ...], int] = defaultdict(int)
    for batch_freq in batch_freqs:
        for token_tuple, count in batch_freq.items():
            total_freq[token_tuple] += count
    
    return dict(total_freq)


def _pretokenize_single_thread_optimized(segments: list[str]) -> dict[tuple[int, ...], int]:
    """单线程优化版本 - 避免进程间通信开销"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex_mod.compile(PAT)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    
    print(f"   使用单线程优化处理...")
    
    # 预编译正则表达式，批量处理
    for chunk in tqdm(segments, desc="预分词进度", unit="段落"):
        # 使用findall而不是finditer，可能更快
        tokens = pat.findall(chunk)
        for token in tokens:
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    
    return dict(freq)


def train_bpe_with_progress(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    带进度条的BPE训练函数
    """
    
    print("=== 带进度条的BPE训练 ===\n")
    
    # 1. 读取文件
    print("📖 读取训练数据...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    file_size = len(text)
    print(f"   文件大小: {file_size:,} 字符")
    
    # 2. 分割特殊token
    print("🔧 处理特殊token...")
    # 将文档分割成独立的段落，每个段落包含一个特殊token
    # 输入"Hello world!<|endoftext|>This is another document.<|endoftext|>Final document here."
    # 输出["Hello world!","This is another document.","Final document here."]
    if not special_tokens:
        segments = [text]
    else:
        escaped = [re.escape(tok) for tok in special_tokens]
        pattern = re.compile("(" + "|".join(escaped) + ")")
        parts = pattern.split(text)
        segments = [seg for seg in parts if seg and seg not in special_tokens]
    
    print(f"   分割成 {len(segments)} 个段落")
    
    # 3. 预分词（优化版本选择）
    print("✂️  预分词...")
    
    # 根据数据量选择优化策略
    if len(segments) < 10000:  # 小数据集用单线程
        freq = _pretokenize_single_thread_optimized(segments)
    else:  # 大数据集用多进程 多进程中通过分batch的形式来获取较快的速度
        freq = _pretokenize_parallel_optimized(segments)
    
    print(f"   预分词完成，共 {len(freq):,} 个唯一token")
    
    # 4. 初始化词汇表
    print("📚 初始化词汇表...")
    id_to_bytes = {}
    bytes_to_id = {}
    
    # 256字节词汇表
    for b in range(256):
        bb = bytes([b])
        id_to_bytes[b] = bb
        bytes_to_id[bb] = b
    
    next_id = 256
    # 特殊token
    for tok in special_tokens:
        b = tok.encode("utf-8")
        id_to_bytes[next_id] = b
        bytes_to_id[b] = next_id
        next_id += 1
    
    print(f"   初始词汇表大小: {len(id_to_bytes)}")
    
    # 5. 准备语料库
    print("📝 准备语料库...")
    # 每一个word对应的ID序列
    corpus_ids = []
    # 每一个word的频率
    corpus_counts = []
    
    for byte_tuple, count in tqdm(freq.items(), desc="转换语料库", unit="token"):
        ids = [bytes_to_id[bytes([b])] for b in byte_tuple]
        corpus_ids.append(ids)
        corpus_counts.append(count)
    
    print(f"   语料库包含 {len(corpus_ids):,} 个token序列")
    
    # 6. 初始化pair计数
    print("🔢 初始化字节对计数...")
    # 这里开始就是开始计算pair了，这里计算的pair是每个token序列内部的pair
    
    def pairs_for_ids(ids):
        ctr = Counter()
        if len(ids) < 2:
            return ctr
        for a, b in zip(ids, ids[1:]):
            ctr[(a, b)] += 1
        return ctr
    
    # 存储每个词的字节对计数
    # 每一个element是Counter对象，key是pair，value是频率，对应这个word中所有pair的频率
    word_pair_counters = []
    # 全局字节对频率统计 通过word的频率来做一个转换
    # key是pair，value是频率
    total_pair_counts = defaultdict(int)
    # 字节对到包含它的词的映射
    # key是pair，value是包含它的词的集合
    pair_to_words = defaultdict(set)
    
    for i, ids in enumerate(tqdm(corpus_ids, desc="计算字节对", unit="序列")):
        ctr = pairs_for_ids(ids)
        word_pair_counters.append(ctr)

        # 这个词的频率
        c = corpus_counts[i]
        for pair, k in ctr.items():
            # 这个字节对的频率 = 这个word的频率 * 这个word中这个字节对的频率
            total_pair_counts[pair] += k * c
            # 这个字节对包含这个词，建立映射关系
            pair_to_words[pair].add(i)
    
    print(f"   发现 {len(total_pair_counts):,} 个唯一字节对")
    
    # 7. 主合并循环
    merges = []
    max_merges = max(0, vocab_size - len(id_to_bytes))
    
    print(f"🔄 开始合并循环，目标: {max_merges:,} 次合并")
    # 一次merge造就一个新的word id

    def merge_pair(a, b, new_token):
        target = (a, b)
        # 找到包含这个pair的所有的word
        affected_words = list(pair_to_words.get(target, set()))
        
        # 遍历所有的words
        for i in affected_words:

            # 找到这个words对应的ID序列 比如 hello —> [104, 101, 108]
            ids = corpus_ids[i]
            if len(ids) < 2:
                continue
            
            # 移除旧的pair计数
            # 拿出之前这个word所包含的所有的pairs
            old_pairs = word_pair_counters[i]

            # 拿到这个word的频率次数
            count_multiplier = corpus_counts[i]

            # 遍历这个word的所有的pairs
            for pair, k in old_pairs.items():
                # 更新全局字节对频率统计
                total_pair_counts[pair] -= k * count_multiplier
                # 如果这个pair的频率小于等于0，则从全局字典中删除这个pair
                if total_pair_counts[pair] <= 0:
                    total_pair_counts.pop(pair, None)
            
            # 执行合并

            # 当前处理的位置
            j = 0
            # 新的ID序列
            out = []
            while j < len(ids):
                # 如果当前位置的ID和要合并的ID相同，则替换为新的ID
                # 在这里嵌入新的ID， 然后跳过两个位置
                if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                    out.append(new_token)
                    j += 2
                else:
                    # 否则，直接添加当前位置的原来的ID
                    out.append(ids[j])
                    j += 1

            # 更新这个word的ID序列
            corpus_ids[i] = out
            
            # 重新计算pairs
            new_ctr = pairs_for_ids(out)
            word_pair_counters[i] = new_ctr
            
            # 更新索引
            for pair in old_pairs.keys():
                # 如果这个pair不在新的ID序列中
                if pair not in new_ctr:
                    # 拿到这个pair包含的所有的words
                    s = pair_to_words.get(pair)
                    if s is not None:
                        # 从这个pair包含的所有的words中删除这个word
                        s.discard(i)
                        if not s: # 如果没有词包含这个字节对了
                            # 从全局字典中删除这个pair
                            pair_to_words.pop(pair, None)
            
            # 更新新的部分
            for pair, k in new_ctr.items():
                # 这个字节对现有的次数 + 这个词贡献的次数
                total_pair_counts[pair] = total_pair_counts.get(pair, 0) + k * count_multiplier
                # 更新pair到word
                s = pair_to_words.get(pair)
                if s is None:
                    pair_to_words[pair] = {i}
                else:
                    s.add(i)

    
    # merge的本质目标就是去更新所有的之前的存储
    def merge_pair_debug(a, b, new_token):
        nonlocal total_pair_counts, pair_to_words, corpus_ids, word_pair_counters
        global merge_step_counter
        merge_step_counter += 1
        DEBUG = True
        
        # 辅助函数，用于打印可读的pair
        def pretty_print_pair(pair):
            id1, id2 = pair
            # 使用 .get 避免在查找特殊 token 时出错
            byte1 = id_to_bytes.get(id1, b'??')
            byte2 = id_to_bytes.get(id2, b'??')
            return f"({repr(byte1)}, {repr(byte2)})"

        initial_len = len(total_pair_counts)
        # --- 审计：记录合并前的所有 unique pairs ---
        initial_pairs_set = set(total_pair_counts.keys())

        if DEBUG and merge_step_counter <= 10:
            print(f"\n" + "="*80)
            print(f"[DEBUG] --- MERGE STEP {merge_step_counter} ---")
            print(f"[DEBUG] Merging pair: {pretty_print_pair((a,b))} -> ID {new_token}")
            print(f"[DEBUG] Initial len(total_pair_counts): {initial_len}")

        affected_words_indices = list(pair_to_words.get((a, b), set()))
        
        # --- 阶段一：更新所有受影响的词的ID序列 ---
        for i in affected_words_indices:
            ids = corpus_ids[i]
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == a and ids[j+1] == b:
                    new_ids.append(new_token)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            corpus_ids[i] = new_ids
            # 注意：我们只更新 corpus_ids，per-word counters 会在重建时自动更新

        # --- 阶段二：完全基于更新后的状态，重建全局计数和索引 ---
        total_pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)
        # 注意：这里的 word_pair_counters 也需要重新计算
        for i in range(len(corpus_ids)):
            # 重新计算 per-word counter
            ctr = pairs_for_ids(corpus_ids[i])
            word_pair_counters[i] = ctr
            
            # 累加到全局计数
            c = corpus_counts[i]
            for pair, k in ctr.items():
                total_pair_counts[pair] += k * c
                pair_to_words[pair].add(i)

        final_len = len(total_pair_counts)
        
        # --- 审计：比较合并前后的 unique pairs 集合 ---
        final_pairs_set = set(total_pair_counts.keys())
        
        removed_pairs = initial_pairs_set - final_pairs_set
        added_pairs = final_pairs_set - initial_pairs_set

        if DEBUG and merge_step_counter <= 10:
            # 将 set 转换为 list 并排序，让输出更稳定
            removed_list = sorted(list(removed_pairs))
            added_list = sorted(list(added_pairs))

            print(f"    (-) Pairs REMOVED ({len(removed_list)} unique types): {{ {', '.join(pretty_print_pair(p) for p in removed_list)} }}")
            print(f"    (+) Pairs ADDED   ({len(added_list)} unique types): {{ {', '.join(pretty_print_pair(p) for p in added_list)} }}")
            
            print(f"[DEBUG] Final len(total_pair_counts): {final_len}")
            print(f"[DEBUG] Net change in len for this step: {final_len - initial_len}  ({len(added_list)} added - {len(removed_list)} removed)")
            print("="*80)
    
    # 使用tqdm显示合并进度
    with tqdm(total=max_merges, desc="合并进度", unit="合并") as pbar:
        for merge_step in range(max_merges):
            if not total_pair_counts:
                pbar.set_description("合并完成 (无更多字节对)")
                break
            
            # 选择最频繁的字节对
            # 这样可以保证字典序最小的pair被选中，同时比较的是实际字节的内容，而不是ID数值
            def pair_key(p):
                a, b = p
                return (total_pair_counts[p], id_to_bytes[a], id_to_bytes[b])
            
            best_pair = max(total_pair_counts.keys(), key=pair_key)
            a, b = best_pair
            
            # 创建新token
            bytes_a = id_to_bytes[a]
            bytes_b = id_to_bytes[b]
            # 创建新bytes，这里就是合并两个bytes
            new_bytes = bytes_a + bytes_b
            new_id = next_id
            next_id += 1
            # 更新词汇表
            id_to_bytes[new_id] = new_bytes
            
            # 执行合并,更新之前设计的所有的数据结构
            merge_pair(a, b, new_id)
            # 记录merge
            merges.append((bytes_a, bytes_b))
            
            # 更新进度条
            pbar.set_postfix({
                '当前合并': f"{bytes_a} + {bytes_b}",
                '词汇表大小': len(id_to_bytes),
                '剩余字节对': len(total_pair_counts)
            })
            pbar.update(1)
            
            if len(id_to_bytes) >= vocab_size:
                pbar.set_description("合并完成 (达到目标词汇表大小)")
                break
    
    print(f"\n✅ 训练完成！")
    print(f"   最终词汇表大小: {len(id_to_bytes):,}")
    print(f"   合并规则数量: {len(merges):,}")
    
    return id_to_bytes, merges


def detect_dataset(input_path: str) -> str:
    """检测数据集类型"""
    path = Path(input_path)
    if "tinystories" in path.name.lower():
        return "tinystories"
    elif "owt" in path.name.lower() or "openwebtext" in path.name.lower():
        return "owt"
    else:
        return "unknown"


def get_dataset_config(dataset_type: str) -> dict:
    """获取数据集配置"""
    configs = {
        "tinystories": {
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"],
            "description": "TinyStories数据集"
        },
        "owt": {
            "vocab_size": 32000,
            "special_tokens": ["<|endoftext|>"],
            "description": "OpenWebText数据集"
        },
        "unknown": {
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"],
            "description": "未知数据集"
        }
    }
    return configs.get(dataset_type, configs["unknown"])


def train_bpe_with_dataset(input_path: str, dataset_type: str = None, vocab_size: int = None, special_tokens: list[str] = None):
    """
    在指定数据集上训练BPE分词器（带进度条版本）
    
    Args:
        input_path: 输入文件路径
        dataset_type: 数据集类型 ("tinystories", "owt", "auto")
        vocab_size: 词汇表大小（如果为None则使用默认配置）
        special_tokens: 特殊token列表（如果为None则使用默认配置）
    """
    
    # 自动检测数据集类型
    if dataset_type is None or dataset_type == "auto":
        dataset_type = detect_dataset(input_path)
    
    # 获取数据集配置
    config = get_dataset_config(dataset_type)
    
    # 使用用户提供的参数或默认配置
    if vocab_size is None:
        vocab_size = config["vocab_size"]
    if special_tokens is None:
        special_tokens = config["special_tokens"]
    
    print(f"=== 训练BPE分词器在{config['description']}上（带进度条） ===\n")
    print(f"检测到的数据集类型: {dataset_type}")
    
    # 配置参数
    print(f"配置参数:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  特殊token: {special_tokens}")
    print(f"  输入文件: {input_path}")
    print()
    
    # 检查文件是否存在
    if not Path(input_path).exists():
        print(f"❌ 错误：找不到输入文件 {input_path}")
        return None
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 训练BPE分词器
        vocab, merges = train_bpe_with_progress(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        
        # 记录结束时间
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n📊 训练统计:")
        print(f"   训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
        print(f"   最终词汇表大小: {len(vocab)}")
        print(f"   合并规则数量: {len(merges)}")
        
        # 分析词汇表
        print(f"\n🔍 词汇表分析:")
        
        # 找出最长的token
        longest_token = max(vocab.values(), key=len)
        print(f"   最长token: {repr(longest_token)} (长度: {len(longest_token)})")
        
        try:
            longest_str = longest_token.decode('utf-8', errors='replace')
            print(f"   最长token解码: '{longest_str}'")
        except:
            print(f"   最长token无法解码为UTF-8")
        
        # 统计不同长度的token
        length_counts = {}
        for token_bytes in vocab.values():
            length = len(token_bytes)
            length_counts[length] = length_counts.get(length, 0) + 1
        
        print(f"\n   Token长度分布:")
        for length in sorted(length_counts.keys()):
            print(f"     长度 {length}: {length_counts[length]:,} 个token")
        
        # 显示一些学习到的token
        learned_tokens = {k: v for k, v in vocab.items() if k >= 256}
        print(f"\n   📝 学习到的token示例（前10个）:")
        for i, (token_id, token_bytes) in enumerate(sorted(learned_tokens.items())[:10]):
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
                print(f"     {token_id}: {repr(token_bytes)} -> '{token_str}'")
            except:
                print(f"     {token_id}: {repr(token_bytes)}")
        
        # 序列化结果到磁盘
        print(f"\n💾 保存结果...")
        output_dir = Path("bpe_results")
        output_dir.mkdir(exist_ok=True)
        
        # 根据数据集类型生成文件名
        dataset_prefix = dataset_type if dataset_type != "unknown" else "custom"
        
        # 保存词汇表（JSON格式）
        vocab_json = {str(k): list(v) for k, v in vocab.items()}
        vocab_file = output_dir / f"{dataset_prefix}_vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, indent=2, ensure_ascii=False)
        
        # 保存合并规则（文本格式）
        merges_file = output_dir / f"{dataset_prefix}_merges.txt"
        with open(merges_file, 'w', encoding='utf-8') as f:
            for left, right in merges:
                f.write(f"{left} {right}\n")
        
        print(f"   结果已保存到:")
        print(f"     📄 词汇表: {vocab_file}")
        print(f"     📄 合并规则: {merges_file}")
        
        # 验证结果
        print(f"\n✅ 验证结果:")
        special_token_bytes = '<|endoftext|>'.encode('utf-8')
        has_special_token = any(token == special_token_bytes for token in vocab.values())
        print(f"   特殊token <|endoftext|> 在词汇表中: {has_special_token}")
        print(f"   词汇表大小符合要求: {len(vocab) == vocab_size}")
        print(f"   合并规则数量: {len(merges)}")
        
        return {
            'vocab': vocab,
            'merges': merges,
            'training_time': training_time,
            'vocab_size': vocab_size,
            'special_tokens': special_tokens,
            'dataset_type': dataset_type
        }
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_bpe_tinystories_with_progress():
    """
    在TinyStories数据集上训练BPE分词器（带进度条版本）
    """
    input_path = "../../data/TinyStoriesV2-GPT4-train.txt"
    return train_bpe_with_dataset(input_path, dataset_type="tinystories")


def train_bpe_owt_with_progress():
    """
    在OpenWebText数据集上训练BPE分词器（带进度条版本）
    """
    input_path = "../../data/owt_train.txt"  # 假设的OWT路径
    return train_bpe_with_dataset(input_path, dataset_type="owt")


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        if dataset_type == "tinystories":
            results = train_bpe_tinystories_with_progress()
        elif dataset_type == "owt":
            results = train_bpe_owt_with_progress()
        elif dataset_type == "auto":
            # 自动检测数据集
            input_path = sys.argv[2] if len(sys.argv) > 2 else "./data/TinyStoriesV2-GPT4-train.txt"
            results = train_bpe_with_dataset(input_path, dataset_type="auto")
        else:
            # 自定义数据集
            input_path = sys.argv[1]
            vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
            results = train_bpe_with_dataset(input_path, vocab_size=vocab_size)
    else:
        # 默认训练TinyStories
        print("使用方法:")
        print("  python train_bpe_with_progress.py tinystories")
        print("  python train_bpe_with_progress.py owt")
        print("  python train_bpe_with_progress.py auto <input_path>")
        print("  python train_bpe_with_progress.py <input_path> [vocab_size]")
        print()
        print("默认训练TinyStories数据集...")
        results = train_bpe_tinystories_with_progress()
    
    if results:
        print("\n🎉 训练完成！")
    else:
        print("\n💥 训练失败！")
