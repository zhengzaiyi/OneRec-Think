"""
Pause Token: implicit reasoning via learnable pause tokens.

Ref: Goyal et al. "Think Before You Speak: Training Language Models With Pause Tokens"

The model learns special <|thought|> token embeddings inserted between the query
and the answer. These tokens give the model extra computation steps at inference,
acting as "thinking time" within the standard forward pass.

Usage:
    from ImpCoT.pause import setup, format_response, PAUSE_TOKEN

    token_id = setup(tokenizer, model)
    assistant_content = format_response(groundtruth, num_pause=5)
"""

PAUSE_TOKEN = "<|thought|>"
DEFAULT_NUM_PAUSE = 5


def setup(tokenizer, model, local_rank=0):
    """Add the pause token to the tokenizer and resize model embeddings.

    Args:
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model (before LoRA wrapping).
        local_rank: For distributed logging.

    Returns:
        int: Token id of the newly added pause token.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [PAUSE_TOKEN]},
        replace_additional_special_tokens=False,
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        if local_rank == 0:
            print(f"[Pause] Added {PAUSE_TOKEN}, vocab size: {len(tokenizer)}")
    return tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)


def format_response(groundtruth, num_pause=DEFAULT_NUM_PAUSE):
    """Format assistant response with pause tokens before the answer.

    Returns:
        str: e.g. ``<|thought|><|thought|>...\\ngroundtruth``
    """
    return f"{PAUSE_TOKEN * num_pause}\n{groundtruth}"
