"""
ViForge Sequence Packing & Context Packaging Engine.
"""

from typing import Any, Dict, List
from viforge.utils.logging import logger


class SequencePacker:
    """
    Packs variable-length tokenized sequences into fixed-length blocks (e.g. 4096 tokens)
    with attention boundary masks and position reset per document.
    """

    def __init__(self, max_seq_len: int = 4096, pad_token_id: int = 0, eos_token_id: int = 2):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def pack_tokenized_sequences(
        self,
        tokenized_examples: List[List[int]],
        add_eos: bool = True,
    ) -> Dict[str, Any]:
        packed_input_ids: List[List[int]] = []
        packed_position_ids: List[List[int]] = []
        packed_attention_masks: List[List[int]] = []

        current_block_ids: List[int] = []
        current_block_pos: List[int] = []

        for seq in tokenized_examples:
            tokens = list(seq)
            if add_eos and (not tokens or tokens[-1] != self.eos_token_id):
                tokens.append(self.eos_token_id)

            while len(tokens) > 0:
                available_space = self.max_seq_len - len(current_block_ids)
                if len(tokens) <= available_space:
                    pos_ids = list(range(len(tokens)))
                    current_block_ids.extend(tokens)
                    current_block_pos.extend(pos_ids)
                    tokens = []
                else:
                    chunk = tokens[:available_space]
                    pos_ids = list(range(len(chunk)))
                    current_block_ids.extend(chunk)
                    current_block_pos.extend(pos_ids)

                    packed_input_ids.append(current_block_ids)
                    packed_position_ids.append(current_block_pos)
                    packed_attention_masks.append([1] * self.max_seq_len)

                    current_block_ids = []
                    current_block_pos = []
                    tokens = tokens[available_space:]

        if current_block_ids:
            num_valid = len(current_block_ids)
            padding_len = self.max_seq_len - num_valid
            current_block_ids.extend([self.pad_token_id] * padding_len)
            current_block_pos.extend([0] * padding_len)
            mask = [1] * num_valid + [0] * padding_len

            packed_input_ids.append(current_block_ids)
            packed_position_ids.append(current_block_pos)
            packed_attention_masks.append(mask)

        logger.info(
            f"Packed {len(tokenized_examples)} variable sequences into {len(packed_input_ids)} "
            f"blocks of length {self.max_seq_len}."
        )

        return {
            "input_ids": packed_input_ids,
            "position_ids": packed_position_ids,
            "attention_mask": packed_attention_masks,
            "num_blocks": len(packed_input_ids),
            "total_tokens": len(packed_input_ids) * self.max_seq_len,
        }
