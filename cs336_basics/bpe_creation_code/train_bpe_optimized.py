#!/usr/bin/env python3
"""
训练BPE分词器 - 优化版本
包含多种性能优化：增量更新、堆优化、并行处理等
"""

import time
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
import regex as regex_mod
import re
import heapq
from functools import lru_cache
from cs336_basics.bpe import train_bpe


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
    batch_size = max(1, len(segments) // (num_processes * 4))
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    
    print(f"   使用 {num_processes} 个进程处理 {len(batches)} 个批次...")
    
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
    """单线程优化版本"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex_mod.compile(PAT)
    freq: dict[tuple[int, ...], int] = defaultdict(int)
    
    print(f"   使用单线程优化处理...")
    
    for chunk in tqdm(segments, desc="预分词进度", unit="段落"):
        tokens = pat.findall(chunk)
        for token in tokens:
            bs = token.encode("utf-8")
            freq[tuple(bs)] += 1
    
    return dict(freq)


class OptimizedBPETrainer:
    """优化的BPE训练器类"""
    
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.id_to_bytes = {}
        self.bytes_to_id = {}
        self.next_id = 0
        
        # 优化的数据结构
        self.corpus_ids = []
        self.corpus_counts = []
        self.word_pair_counters = []
        self.total_pair_counts = defaultdict(int)
        self.pair_to_words = defaultdict(set)
        
        # 堆优化：维护最频繁的字节对
        self.pair_heap = []
        self.heap_rebuild_needed = True
        
        # 缓存优化（使用@lru_cache装饰器）
        
    def _build_initial_vocab(self):
        """初始化词汇表"""
        # 256字节词汇表
        for b in range(256):
            bb = bytes([b])
            self.id_to_bytes[b] = bb
            self.bytes_to_id[bb] = b
        
        self.next_id = 256
        # 特殊token
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            self.id_to_bytes[self.next_id] = b
            self.bytes_to_id[b] = self.next_id
            self.next_id += 1
    
    @lru_cache(maxsize=10000)
    def _get_pair_key(self, pair: tuple[int, int]) -> tuple[int, bytes, bytes]:
        """缓存的字节对键计算"""
        a, b = pair
        return (self.total_pair_counts[pair], self.id_to_bytes[a], self.id_to_bytes[b])
    
    def _rebuild_heap(self):
        """重建字节对堆"""
        self.pair_heap = []
        # 只处理频率大于0的字节对
        for pair, freq in self.total_pair_counts.items():
            if freq > 0:
                # 直接使用频率，避免重复计算
                bytes_a = self.id_to_bytes[pair[0]]
                bytes_b = self.id_to_bytes[pair[1]]
                heapq.heappush(self.pair_heap, (-freq, bytes_a, bytes_b, pair))
        self.heap_rebuild_needed = False
    
    def _get_best_pair(self) -> tuple[int, int] | None:
        """获取最频繁的字节对（使用堆优化）"""
        if not self.total_pair_counts:
            return None
        
        if self.heap_rebuild_needed:
            self._rebuild_heap()
        
        # 从堆中获取最频繁的字节对
        while self.pair_heap:
            neg_freq, bytes_a, bytes_b, pair = heapq.heappop(self.pair_heap)
            # 检查频率是否仍然有效（可能在其他地方被更新）
            current_freq = self.total_pair_counts.get(pair, 0)
            if current_freq > 0:
                return pair
        
        return None
    
    def _pairs_for_ids(self, ids: list[int]) -> Counter[tuple[int, int]]:
        """计算ID序列的字节对"""
        ctr = Counter()
        if len(ids) < 2:
            return ctr
        for a, b in zip(ids, ids[1:]):
            ctr[(a, b)] += 1
        return ctr
    
    def _merge_pair_optimized(self, a: int, b: int, new_token: int) -> None:
        """优化的字节对合并函数"""
        target = (a, b)
        affected_words = list(self.pair_to_words.get(target, set()))
        
        if len(affected_words) < 100:  # 小批量用串行处理
            self._merge_pair_serial(a, b, new_token, affected_words)
        else:  # 大批量用并行处理
            self._merge_pair_parallel(a, b, new_token, affected_words)
        
        # 标记需要重建堆
        self.heap_rebuild_needed = True
    
    def _merge_pair_serial(self, a: int, b: int, new_token: int, affected_words: list) -> None:
        """串行合并处理"""
        batch_updates = []
        
        for i in affected_words:
            ids = self.corpus_ids[i]
            if len(ids) < 2:
                continue
            
            old_pairs = self.word_pair_counters[i]
            count_multiplier = self.corpus_counts[i]
            
            # 收集需要移除的字节对
            for pair, k in old_pairs.items():
                self.total_pair_counts[pair] -= k * count_multiplier
                if self.total_pair_counts[pair] <= 0:
                    self.total_pair_counts.pop(pair, None)
            
            # 执行合并
            out = self._perform_merge(ids, a, b, new_token)
            self.corpus_ids[i] = out
            
            # 重新计算字节对
            new_ctr = self._pairs_for_ids(out)
            self.word_pair_counters[i] = new_ctr
            
            # 收集批量更新
            batch_updates.append((i, old_pairs, new_ctr, count_multiplier))
        
        # 批量应用更新
        self._apply_batch_updates(batch_updates)
    
    def _merge_pair_parallel(self, a: int, b: int, new_token: int, affected_words: list) -> None:
        """并行合并处理"""
        # 将受影响的词分批
        batch_size = max(1, len(affected_words) // (cpu_count() * 2))
        word_batches = [affected_words[i:i + batch_size] for i in range(0, len(affected_words), batch_size)]
        
        # 准备并行处理的数据
        merge_tasks = []
        for batch in word_batches:
            task_data = []
            for i in batch:
                if i < len(self.corpus_ids) and len(self.corpus_ids[i]) >= 2:
                    task_data.append({
                        'word_id': i,
                        'ids': self.corpus_ids[i].copy(),
                        'old_pairs': dict(self.word_pair_counters[i]),
                        'count_multiplier': self.corpus_counts[i]
                    })
            if task_data:
                merge_tasks.append((a, b, new_token, task_data))
        
        # 并行处理
        with Pool(processes=min(cpu_count(), len(merge_tasks))) as pool:
            results = pool.map(self._process_merge_batch, merge_tasks)
        
        # 收集结果并应用更新
        all_batch_updates = []
        for batch_results in results:
            all_batch_updates.extend(batch_results)
        
        # 批量应用所有更新
        self._apply_batch_updates(all_batch_updates)
    
    @staticmethod
    def _process_merge_batch(args):
        """处理单个合并批次（静态方法，用于多进程）"""
        a, b, new_token, task_data = args
        batch_updates = []
        
        for task in task_data:
            word_id = task['word_id']
            ids = task['ids']
            old_pairs = task['old_pairs']
            count_multiplier = task['count_multiplier']
            
            # 执行合并
            out = OptimizedBPETrainer._perform_merge_static(ids, a, b, new_token)
            
            # 重新计算字节对
            new_ctr = OptimizedBPETrainer._pairs_for_ids_static(out)
            
            # 收集更新
            batch_updates.append((word_id, old_pairs, new_ctr, count_multiplier))
        
        return batch_updates
    
    def _perform_merge(self, ids: list[int], a: int, b: int, new_token: int) -> list[int]:
        """执行字节对合并"""
        j = 0
        out = []
        while j < len(ids):
            if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                out.append(new_token)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        return out
    
    @staticmethod
    def _perform_merge_static(ids: list[int], a: int, b: int, new_token: int) -> list[int]:
        """静态方法版本的合并（用于多进程）"""
        j = 0
        out = []
        while j < len(ids):
            if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                out.append(new_token)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        return out
    
    @staticmethod
    def _pairs_for_ids_static(ids: list[int]) -> dict:
        """静态方法版本的字节对计算（用于多进程）"""
        ctr = {}
        if len(ids) < 2:
            return ctr
        for a, b in zip(ids, ids[1:]):
            pair = (a, b)
            ctr[pair] = ctr.get(pair, 0) + 1
        return ctr
    
    def _apply_batch_updates(self, batch_updates: list):
        """批量应用索引更新（优化版本）"""
        # 使用集合来跟踪需要清理和添加的字节对
        pairs_to_remove = set()
        pairs_to_add = {}
        
        # 第一遍：收集所有需要清理的字节对
        for i, old_pairs, new_ctr, count_multiplier in batch_updates:
            for pair in old_pairs.keys():
                if pair not in new_ctr:
                    pairs_to_remove.add(pair)
        
        # 第二遍：收集所有需要添加的字节对
        for i, old_pairs, new_ctr, count_multiplier in batch_updates:
            for pair, k in new_ctr.items():
                if pair not in pairs_to_add:
                    pairs_to_add[pair] = []
                pairs_to_add[pair].append((i, k * count_multiplier))
        
        # 批量清理旧索引
        for pair in pairs_to_remove:
            s = self.pair_to_words.get(pair)
            if s is not None:
                # 批量移除所有相关的词
                for i, old_pairs, new_ctr, count_multiplier in batch_updates:
                    s.discard(i)
                if not s:
                    self.pair_to_words.pop(pair, None)
        
        # 批量添加新索引
        for pair, updates in pairs_to_add.items():
            # 计算总频率
            total_freq = sum(freq for _, freq in updates)
            self.total_pair_counts[pair] = self.total_pair_counts.get(pair, 0) + total_freq
            
            # 添加词索引
            s = self.pair_to_words.get(pair)
            if s is None:
                s = set()
                self.pair_to_words[pair] = s
            
            for i, _ in updates:
                s.add(i)
    
    def train(self, input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """训练BPE分词器"""
        print("=== 优化版BPE训练 ===\n")
        
        # 1. 读取文件
        print("📖 读取训练数据...")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        file_size = len(text)
        print(f"   文件大小: {file_size:,} 字符")
        
        # 2. 分割特殊token
        print("🔧 处理特殊token...")
        if not self.special_tokens:
            segments = [text]
        else:
            escaped = [re.escape(tok) for tok in self.special_tokens]
            pattern = re.compile("(" + "|".join(escaped) + ")")
            parts = pattern.split(text)
            segments = [seg for seg in parts if seg and seg not in self.special_tokens]
        
        print(f"   分割成 {len(segments)} 个段落")
        
        # 3. 预分词
        print("✂️  预分词...")
        if len(segments) < 10000:
            freq = _pretokenize_single_thread_optimized(segments)
        else:
            freq = _pretokenize_parallel_optimized(segments)
        
        print(f"   预分词完成，共 {len(freq):,} 个唯一token")
        
        # 4. 初始化词汇表
        print("📚 初始化词汇表...")
        self._build_initial_vocab()
        print(f"   初始词汇表大小: {len(self.id_to_bytes)}")
        
        # 5. 准备语料库
        print("📝 准备语料库...")
        for byte_tuple, count in tqdm(freq.items(), desc="转换语料库", unit="token"):
            ids = [self.bytes_to_id[bytes([b])] for b in byte_tuple]
            self.corpus_ids.append(ids)
            self.corpus_counts.append(count)
        
        print(f"   语料库包含 {len(self.corpus_ids):,} 个token序列")
        
        # 6. 初始化字节对计数
        print("🔢 初始化字节对计数...")
        for i, ids in enumerate(tqdm(self.corpus_ids, desc="计算字节对", unit="序列")):
            ctr = self._pairs_for_ids(ids)
            self.word_pair_counters.append(ctr)
            c = self.corpus_counts[i]
            for pair, k in ctr.items():
                self.total_pair_counts[pair] += k * c
                self.pair_to_words[pair].add(i)
        
        print(f"   发现 {len(self.total_pair_counts):,} 个唯一字节对")
        
        # 7. 主合并循环
        merges = []
        max_merges = max(0, self.vocab_size - len(self.id_to_bytes))
        
        print(f"🔄 开始合并循环，目标: {max_merges:,} 次合并")
        
        # 性能统计
        merge_times = []
        parallel_count = 0
        serial_count = 0
        
        with tqdm(total=max_merges, desc="合并进度", unit="合并") as pbar:
            for merge_step in range(max_merges):
                merge_start = time.time()
                
                best_pair = self._get_best_pair()
                if best_pair is None:
                    pbar.set_description("合并完成 (无更多字节对)")
                    break
                
                a, b = best_pair
                
                # 创建新token
                bytes_a = self.id_to_bytes[a]
                bytes_b = self.id_to_bytes[b]
                new_bytes = bytes_a + bytes_b
                new_id = self.next_id
                self.next_id += 1
                self.id_to_bytes[new_id] = new_bytes
                
                # 执行合并
                affected_count = len(self.pair_to_words.get((a, b), set()))
                if affected_count >= 100:
                    parallel_count += 1
                else:
                    serial_count += 1
                
                self._merge_pair_optimized(a, b, new_id)
                merges.append((bytes_a, bytes_b))
                
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)
                
                # 更新进度条
                avg_time = sum(merge_times[-10:]) / min(10, len(merge_times))
                pbar.set_postfix({
                    '当前合并': f"{bytes_a} + {bytes_b}",
                    '词汇表大小': len(self.id_to_bytes),
                    '剩余字节对': len(self.total_pair_counts),
                    '影响词数': affected_count,
                    '平均时间': f"{avg_time:.3f}s"
                })
                pbar.update(1)
                
                if len(self.id_to_bytes) >= self.vocab_size:
                    pbar.set_description("合并完成 (达到目标词汇表大小)")
                    break
        
        # 输出性能统计
        if merge_times:
            print(f"\n📊 合并性能统计:")
            print(f"   总合并次数: {len(merge_times)}")
            print(f"   并行处理次数: {parallel_count}")
            print(f"   串行处理次数: {serial_count}")
            print(f"   平均合并时间: {sum(merge_times) / len(merge_times):.3f} 秒")
            print(f"   最快合并时间: {min(merge_times):.3f} 秒")
            print(f"   最慢合并时间: {max(merge_times):.3f} 秒")
        
        print(f"\n✅ 训练完成！")
        print(f"   最终词汇表大小: {len(self.id_to_bytes):,}")
        print(f"   合并规则数量: {len(merges):,}")
        
        return self.id_to_bytes, merges


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


def train_bpe_optimized(input_path: str, dataset_type: str = None, vocab_size: int = None, special_tokens: list[str] = None):
    """
    优化的BPE训练函数
    
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
    
    print(f"=== 优化版BPE训练器在{config['description']}上 ===\n")
    print(f"检测到的数据集类型: {dataset_type}")
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
        # 创建优化训练器
        trainer = OptimizedBPETrainer(vocab_size, special_tokens)
        
        # 训练BPE分词器
        vocab, merges = trainer.train(input_path)
        
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
        vocab_file = output_dir / f"{dataset_prefix}_vocab_optimized.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, indent=2, ensure_ascii=False)
        
        # 保存合并规则（文本格式）
        merges_file = output_dir / f"{dataset_prefix}_merges_optimized.txt"
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


def train_bpe_tinystories_optimized():
    """在TinyStories数据集上训练BPE分词器（优化版本）"""
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    return train_bpe_optimized(input_path, dataset_type="tinystories")


def train_bpe_owt_optimized():
    """在OpenWebText数据集上训练BPE分词器（优化版本）"""
    input_path = "./data/owt_train.txt"
    return train_bpe_optimized(input_path, dataset_type="owt")


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        if dataset_type == "tinystories":
            results = train_bpe_tinystories_optimized()
        elif dataset_type == "owt":
            results = train_bpe_owt_optimized()
        elif dataset_type == "auto":
            # 自动检测数据集
            input_path = sys.argv[2] if len(sys.argv) > 2 else "./data/TinyStoriesV2-GPT4-train.txt"
            results = train_bpe_optimized(input_path, dataset_type="auto")
        else:
            # 自定义数据集
            input_path = sys.argv[1]
            vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
            results = train_bpe_optimized(input_path, vocab_size=vocab_size)
    else:
        # 默认训练TinyStories
        print("使用方法:")
        print("  python train_bpe_optimized.py tinystories")
        print("  python train_bpe_optimized.py owt")
        print("  python train_bpe_optimized.py auto <input_path>")
        print("  python train_bpe_optimized.py <input_path> [vocab_size]")
        print()
        print("默认训练TinyStories数据集...")
        results = train_bpe_tinystories_optimized()
    
    if results:
        print("\n🎉 优化版训练完成！")
    else:
        print("\n💥 优化版训练失败！")
