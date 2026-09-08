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

    @staticmethod
    def create_block_diagonal_causal_mask(
        document_ids: List[int], max_seq_len: int
    ) -> List[List[bool]]:
        """
        Constructs a 2D block-diagonal causal attention mask to prevent cross-document leakage.
        Position (i, j) is True (attended) if:
        1. Both i and j belong to the same non-padding document (doc_ids[i] == doc_ids[j] >= 0)
        2. j <= i (causal autoregressive constraint)
        """
        mask = [[False] * max_seq_len for _ in range(max_seq_len)]
        for i in range(max_seq_len):
            doc_i = document_ids[i] if i < len(document_ids) else -1
            if doc_i < 0:
                continue
            for j in range(i + 1):
                doc_j = document_ids[j] if j < len(document_ids) else -1
                if doc_i == doc_j and doc_j >= 0:
                    mask[i][j] = True
        return mask

    def pack_tokenized_sequences(
        self,
        tokenized_examples: List[List[int]],
        add_eos: bool = True,
        build_block_diagonal_mask: bool = False,
    ) -> Dict[str, Any]:
        packed_input_ids: List[List[int]] = []
        packed_position_ids: List[List[int]] = []
        packed_attention_masks: List[List[int]] = []
        packed_document_ids: List[List[int]] = []
        packed_cu_seqlens: List[List[int]] = []
        packed_block_diagonal_masks: List[List[List[bool]]] = []

        current_block_ids: List[int] = []
        current_block_pos: List[int] = []
        current_block_doc: List[int] = []
        current_block_cu: List[int] = [0]

        for doc_idx, seq in enumerate(tokenized_examples):
            tokens = list(seq)
            if add_eos and (not tokens or tokens[-1] != self.eos_token_id):
                tokens.append(self.eos_token_id)

            while len(tokens) > 0:
                available_space = self.max_seq_len - len(current_block_ids)
                if len(tokens) <= available_space:
                    pos_ids = list(range(len(tokens)))
                    current_block_ids.extend(tokens)
                    current_block_pos.extend(pos_ids)
                    current_block_doc.extend([doc_idx] * len(tokens))
                    current_block_cu.append(len(current_block_ids))
                    tokens = []
                else:
                    chunk = tokens[:available_space]
                    pos_ids = list(range(len(chunk)))
                    current_block_ids.extend(chunk)
                    current_block_pos.extend(pos_ids)
                    current_block_doc.extend([doc_idx] * len(chunk))
                    current_block_cu.append(len(current_block_ids))

                    packed_input_ids.append(current_block_ids)
                    packed_position_ids.append(current_block_pos)
                    packed_attention_masks.append([1] * self.max_seq_len)
                    packed_document_ids.append(current_block_doc)
                    packed_cu_seqlens.append(current_block_cu)
                    if build_block_diagonal_mask:
                        packed_block_diagonal_masks.append(
                            self.create_block_diagonal_causal_mask(
                                current_block_doc, self.max_seq_len
                            )
                        )

                    current_block_ids = []
                    current_block_pos = []
                    current_block_doc = []
                    current_block_cu = [0]
                    tokens = tokens[available_space:]

        if current_block_ids:
            num_valid = len(current_block_ids)
            padding_len = self.max_seq_len - num_valid
            current_block_ids.extend([self.pad_token_id] * padding_len)
            current_block_pos.extend([0] * padding_len)
            current_block_doc.extend([-1] * padding_len)
            mask = [1] * num_valid + [0] * padding_len

            packed_input_ids.append(current_block_ids)
            packed_position_ids.append(current_block_pos)
            packed_attention_masks.append(mask)
            packed_document_ids.append(current_block_doc)
            packed_cu_seqlens.append(current_block_cu)
            if build_block_diagonal_mask:
                packed_block_diagonal_masks.append(
                    self.create_block_diagonal_causal_mask(current_block_doc, self.max_seq_len)
                )

        logger.info(
            f"Packed {len(tokenized_examples)} variable sequences into {len(packed_input_ids)} "
            f"blocks of length {self.max_seq_len} with document isolation."
        )

        result: Dict[str, Any] = {
            "input_ids": packed_input_ids,
            "position_ids": packed_position_ids,
            "attention_mask": packed_attention_masks,
            "document_ids": packed_document_ids,
            "cu_seqlens": packed_cu_seqlens,
            "num_blocks": len(packed_input_ids),
            "total_tokens": len(packed_input_ids) * self.max_seq_len,
        }
        if build_block_diagonal_mask:
            result["block_diagonal_masks"] = packed_block_diagonal_masks

        return result

