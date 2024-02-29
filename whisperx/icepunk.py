import difflib
from logging import getLogger
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizerFast,
)

log = getLogger(__name__)

HF_TOKEN = os.environ["HF_TOKEN"]


class IcePunk:
    def __init__(
        self,
        punk_path: str = "mideind/IcePunk",
        punk_batch_size: int = 20,  # These batch sizes using fp16 use about <10GB of memory
    ):
        self.punk_batch_size = punk_batch_size
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch_dtype = torch.float16
        else:
            device = torch.device("cpu")
            torch_dtype = torch.float32
        log.info(f"Using device: {device}")
        self.device = device
        self.torch_dtype = torch_dtype

        # Load the IcePunk model
        self.punk_model = (
            MBartForConditionalGeneration.from_pretrained(punk_path, token=HF_TOKEN)
            .eval()
            .to(self.torch_dtype)
            .to(self.device)
        )
        self.punk_tokenizer = MBartTokenizerFast.from_pretrained(punk_path, token=HF_TOKEN)

    def postprocess(self, asr_result, num_tokens = 600):
        # First we tokenize the text - so we can count the tokens and split it into segments
        all_tokens = self.punk_tokenizer.encode(asr_result, add_special_tokens=False)
        log.info(f"Token count: {len(all_tokens)}")
        # Now we chunk the tokens into segments, each segment starts with at most MAX_TOKENS tokens
        segments = []
        for i in range(0, len(all_tokens), num_tokens):
            segments.append(all_tokens[i : i + num_tokens])

        # Now we add 50 tokens from the end of the previous segment to the start of the next segment
        for i in range(1, len(segments)):
            segments[i] = segments[i - 1][-50:] + segments[i]
        for i in range(len(segments)):
            # Add the EOS and src_lang tokens
            segments[i] = segments[i] + [2, 250002]
        # Convert to a tensor - then all segments need to be of equal length, so we pad
        max_len = max(len(s) for s in segments)
        pad_id = self.punk_tokenizer.pad_token_id
        padded_segments = [s + [pad_id] * (max_len - len(s)) for s in segments]
        input_ids = torch.tensor(padded_segments, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()

        # Now we run these segments through the model.generate method
        # We batch the segments to avoid OOM errors
        punk_segments = []
        punk_start_time = time.time()
        for i in range(0, len(segments), self.punk_batch_size):
            log.info(f"Punk model on segments {i} to {i+self.punk_batch_size}")
            outputs = self.punk_model.generate(
                inputs=input_ids[i : i + self.punk_batch_size].to(self.device),
                attention_mask=attention_mask[i : i + self.punk_batch_size].to(
                    self.device
                ),
                decoder_start_token_id=250001,
                max_new_tokens=1024,
            )
            for g in outputs:
                punk_segments.append(
                    self.punk_tokenizer.decode(g, skip_special_tokens=True)
                )
        punk_end_time = time.time()
        log.info(f"IcePunk took {punk_end_time - punk_start_time} seconds")
        log.info(
            f"Tokens/second: {len(all_tokens) / (punk_end_time - punk_start_time)}"
        )
        # Now we need to join the segments together using some heuristics
        punctuated_text = stitch_overlapping_segments(
            punk_segments, subword_token_overlap=50, chars_per_subword=2
        )

        log.debug("FINAL:")
        log.debug(punctuated_text)

        return punctuated_text

def stitch_overlapping_segments(segments: List[str], subword_token_overlap: int, chars_per_subword: int = 2) -> str:
    if len(segments) == 0:
        return ""
    if len(segments) == 1:
        return segments[0]
    # Stitched tracks the complete stitched-together transcript
    stitched = segments[0]
    for i in range(1, len(segments)):
        # Find the longest common substring between the last segment and the current one
        curr = segments[i]
        matcher = difflib.SequenceMatcher(None, stitched, curr, autojunk=False)
        # We search for the longest common substring in the subword token overlap
        # We need to convert the subword token overlap to characters
        overlap_in_chars = subword_token_overlap * chars_per_subword
        match = matcher.find_longest_match(
            alo=-overlap_in_chars, ahi=len(stitched), blo=0, bhi=overlap_in_chars
        )
        # If we have a match, we stitch the segments together
        if match.size > 0:
            stitched = stitched[: match.a] + curr[match.b :]
        else:
            log.error(
                "No overlap between segments, this should not happen, please investigate. I will continue"
            )
            log.error(f"Stitched so far: {stitched}")
            log.error(f"Current segment: {curr}")
            log.error(f"{overlap_in_chars=}")
            stitched += " " + curr
    return stitched