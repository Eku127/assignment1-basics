"""
BPE Tokenizer Class

This module implements the Tokenizer class for encoding and decoding text using BPE.

Features:
- Encode text to token IDs
- Decode token IDs to text
- Handle special tokens
- Memory-efficient streaming with encode_iterable
- Load from serialized vocabulary and merges
"""

import ast
import json
from typing import Iterable, Iterator, Optional

from cs336_basics.bpe.utils import pretokenize_string


class Tokenizer:
    """
    BPE Tokenizer for encoding/decoding text.
    
    The tokenizer uses a vocabulary and merge rules learned during BPE training
    to efficiently encode text into token IDs and decode them back.
    
    Example:
        >>> vocab = {i: bytes([i]) for i in range(256)}  # Byte vocabulary
        >>> vocab[256] = b'th'
        >>> merges = [(b't', b'h')]
        >>> tokenizer = Tokenizer(vocab, merges)
        >>> ids = tokenizer.encode("the")
        >>> text = tokenizer.decode(ids)
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None
    ):
        """
        Construct a tokenizer from vocabulary, merges, and special tokens.
        
        Args:
            vocab: Vocabulary mapping token IDs to byte sequences
            merges: Ordered list of BPE merge rules
            special_tokens: Optional list of special token strings
        """
        # 1. Store basic data (make copies to avoid mutation)
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Pre-sort special tokens by length (descending) for efficient matching
        # This avoids re-sorting on every encode() call
        self._sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
        # 2. Add special tokens to vocab if not present
        if special_tokens:
            next_id = max(vocab.keys()) + 1
            for special_token in special_tokens:
                special_bytes = special_token.encode('utf-8')
                if special_bytes not in self.vocab.values():
                    self.vocab[next_id] = special_bytes
                    next_id += 1
        
        # 3. Build reverse mapping
        self.id_to_bytes: dict[int, bytes] = {}
        self.bytes_to_id: dict[bytes, int] = {}
        
        for token_id, token_bytes in self.vocab.items():
            self.id_to_bytes[token_id] = token_bytes
            self.bytes_to_id[token_bytes] = token_id
        
        # 4. Build merge rules mapping for efficient lookup
        self.merges_map: dict[tuple[bytes, bytes], bytes] = {}
        for a, b in merges:
            merged_bytes = a + b
            self.merges_map[(a, b)] = merged_bytes
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None
    ):
        """
        Load tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: Optional list of special token strings
        
        Returns:
            Tokenizer instance
        
        Example:
            >>> tokenizer = Tokenizer.from_files(
            ...     "vocab.json",
            ...     "merges.txt",
            ...     special_tokens=["<|endoftext|>"]
            ... )
        """
        # Load vocabulary (JSON: string IDs -> list of byte values)
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        vocab: dict[int, bytes] = {int(k): bytes(v) for k, v in raw_vocab.items()}
        
        # Load merges (text file: lines like "b'th' b'e'")
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Handle format like "b' ' b'p'" or "b' p' b'a'"
                if line.startswith("b'") and " b'" in line:
                    # Find second b' position
                    second_b_pos = line.find(" b'")
                    if second_b_pos != -1:
                        left_part = line[:second_b_pos]
                        right_part = line[second_b_pos + 1:]  # Skip leading space
                    else:
                        continue
                else:
                    # Handle normal format "b'th' b'e'"
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    left_part = parts[0]
                    right_part = parts[1]
                
                # Use ast.literal_eval to convert b'...' repr back to bytes
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
        Encode input text to token ID sequence.
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        
        Example:
            >>> tokenizer.encode("hello world")
            [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
        """
        # Fast path: if no special tokens, directly encode
        if not self.special_tokens:
            return self._encode_text(text)
        
        result_ids = []
        remaining_text = text
        
        # Handle special tokens by finding them first
        while remaining_text:
            # Find earliest occurring special token (prefer longer matches)
            earliest_special = None
            earliest_pos = len(remaining_text)
            
            # Use pre-sorted special tokens (cached in __init__)
            for special_token in self._sorted_special_tokens:
                pos = remaining_text.find(special_token)
                if pos != -1 and pos < earliest_pos:
                    earliest_special = special_token
                    earliest_pos = pos
            
            if earliest_special is not None:
                # Process text before special token
                if earliest_pos > 0:
                    before_text = remaining_text[:earliest_pos]
                    before_ids = self._encode_text(before_text)
                    result_ids.extend(before_ids)
                
                # Add special token
                special_id = self.bytes_to_id.get(earliest_special.encode('utf-8'))
                if special_id is not None:
                    result_ids.append(special_id)
                
                # Update remaining text
                remaining_text = remaining_text[earliest_pos + len(earliest_special):]
            else:
                # No more special tokens, process remaining text
                remaining_ids = self._encode_text(remaining_text)
                result_ids.extend(remaining_ids)
                break
        
        return result_ids
    
    def _encode_text(self, text: str) -> list[int]:
        """
        Encode normal text (without special tokens).
        
        Args:
            text: Input text without special tokens
        
        Returns:
            List of token IDs
        """
        # 1. Pre-tokenize using GPT-2 regex pattern
        pretokens = pretokenize_string(text)
        
        # 2. Apply BPE merges to each pre-token separately
        all_merged_tokens = []
        for token in pretokens:
            # Convert token to byte tuple (each byte as separate bytes object)
            byte_tuple = self._to_bytes_tuple(token)
            # Apply merges to this token
            merged_token = self._apply_merges(list(byte_tuple))
            all_merged_tokens.extend(merged_token)
        
        # 3. Convert bytes to IDs
        ids = self._bytes_to_ids(all_merged_tokens)
        
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        
        Lazily yields token IDs without loading all text into memory.
        Useful for tokenizing large files.
        
        Args:
            iterable: Iterable of strings (e.g., file handle)
        
        Yields:
            Token IDs one at a time
        
        Example:
            >>> with open("large_file.txt") as f:
            ...     for token_id in tokenizer.encode_iterable(f):
            ...         process(token_id)
        """
        for text in iterable:
            tokens = self.encode(text)
            for token_id in tokens:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode token ID sequence to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text string
        
        Example:
            >>> tokenizer.decode([104, 101, 108, 108, 111])
            'hello'
        
        Note:
            Invalid UTF-8 sequences are replaced with U+FFFD replacement character.
        """
        # 1. Map token IDs to byte sequences
        bytes_tokens = self._ids_to_bytes(ids)
        
        # 2. Concatenate all bytes
        concat_bytes = b''.join(bytes_tokens)
        
        # 3. Decode to Unicode string (replace invalid sequences)
        text = concat_bytes.decode('utf-8', errors='replace')
        
        return text
    
    def _to_bytes_tuple(self, word: str) -> tuple[bytes, ...]:
        """
        Convert string to tuple of individual bytes.
        
        Args:
            word: Input string
        
        Returns:
            Tuple of bytes objects (one per byte)
        """
        byte_list = list(word.encode("utf-8"))
        byte_list = [bytes([x]) for x in byte_list]
        return tuple(byte_list)
    
    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """
        Apply BPE merge rules to byte token sequence.
        
        Merges are applied in order of creation (as learned during training).
        
        Args:
            tokens: List of byte tokens
        
        Returns:
            List of merged byte tokens
        """
        if not tokens or len(tokens) < 2:
            return tokens
        
        out = tokens[:]
        
        # Iterate through merges in order
        for a, b in self.merges:
            merged = self.merges_map.get((a, b))
            if merged is None:
                continue
            
            # Scan left-to-right and merge matching pairs
            i = 0
            while i < len(out) - 1:
                # Check if current pair matches merge rule
                if out[i] == a and out[i + 1] == b:
                    # Merge i and i+1
                    out[i] = merged
                    del out[i + 1]
                    # Back up one position to check new pairs formed
                    if i > 0:
                        i -= 1
                    continue
                i += 1
        
        return out
    
    def _bytes_to_ids(self, tokens: list[bytes]) -> list[int]:
        """
        Convert byte tokens to token IDs.
        
        Args:
            tokens: List of byte tokens
        
        Returns:
            List of token IDs
        
        Raises:
            ValueError: If a token is not in vocabulary
        """
        ids = []
        for token in tokens:
            if token in self.bytes_to_id:
                ids.append(self.bytes_to_id[token])
            else:
                raise ValueError(f"Token {token!r} not found in vocab.")
        return ids
    
    def _ids_to_bytes(self, ids: list[int]) -> list[bytes]:
        """
        Convert token IDs to byte tokens.
        
        Args:
            ids: List of token IDs
        
        Returns:
            List of byte tokens
        
        Raises:
            ValueError: If an ID is not in vocabulary
        """
        tokens = []
        for id in ids:
            if id in self.id_to_bytes:
                tokens.append(self.id_to_bytes[id])
            else:
                raise ValueError(f"ID {id!r} not found in vocab.")
        
        return tokens

