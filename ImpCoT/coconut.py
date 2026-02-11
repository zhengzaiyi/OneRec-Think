"""
Coconut: Chain of Continuous Thought via latent-space reasoning.

Ref: Hao et al. "Training Large Language Models to Reason in a Continuous Latent Space"

Instead of discrete CoT tokens, the model generates "thoughts" as hidden-state
vectors autoregressively in its own latent space.  During training the <thought>
placeholder embeddings are replaced by these continuous representations, and
the LM loss is computed only on the answer tokens that follow <eot>.

Usage:
    from ImpCoT.coconut import setup, format_response, CoconutTrainer
    from ImpCoT.coconut import BOT_TOKEN, THOUGHT_TOKEN, EOT_TOKEN

    token_ids = setup(tokenizer, model)           # before LoRA
    assistant_content = format_response(gt, 5)    # data formatting
    trainer = CoconutTrainer(                     # replaces Trainer
        coconut_token_ids=token_ids, model=model, args=args, ...
    )
"""

import torch
from transformers import Trainer

BOT_TOKEN = "<bot>"
THOUGHT_TOKEN = "<thought>"
EOT_TOKEN = "<eot>"
DEFAULT_NUM_THOUGHTS = 5


# ---------------------------------------------------------------------------
# Setup & data helpers
# ---------------------------------------------------------------------------

def setup(tokenizer, model, local_rank=0):
    """Add Coconut special tokens and resize model embeddings.

    Must be called **before** LoRA wrapping so that the new embeddings are
    included in ``trainable_token_indices``.

    Returns:
        dict: ``{token_str: token_id}`` for ``<bot>``, ``<thought>``, ``<eot>``.
    """
    tokens = [BOT_TOKEN, THOUGHT_TOKEN, EOT_TOKEN]
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens},
        replace_additional_special_tokens=False,
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        if local_rank == 0:
            print(f"[Coconut] Added {tokens}, vocab size: {len(tokenizer)}")
    return {t: tokenizer.convert_tokens_to_ids(t) for t in tokens}


def format_response(groundtruth, num_thoughts=DEFAULT_NUM_THOUGHTS):
    """Format assistant response with Coconut thought structure.

    Returns:
        str: e.g. ``<bot><thought><thought>...<eot>\\ngroundtruth``
    """
    return f"{BOT_TOKEN}{THOUGHT_TOKEN * num_thoughts}{EOT_TOKEN}\n{groundtruth}"


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------

class CoconutTrainer(Trainer):
    """HuggingFace Trainer with Coconut continuous-thought forward pass.

    For each sample the training step is:

    1. Embed ``input_ids`` and locate ``<bot>`` / ``<eot>`` markers.
    2. Forward the prefix up to ``<bot>`` (inclusive); take the last hidden
       state as the first continuous thought.
    3. Autoregressively generate the remaining continuous thoughts in the
       model's latent space.
    4. Concatenate ``[prefix | thoughts | suffix]`` and run a final forward
       pass with the LM-head to compute loss (only on answer tokens).

    Samples within a batch are processed sequentially because the autoregressive
    thought loop prevents simple batching (different ``<bot>``/``<eot>`` offsets
    after tokenisation + padding).
    """

    def __init__(self, coconut_token_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot_id = coconut_token_ids[BOT_TOKEN]
        self.thought_id = coconut_token_ids[THOUGHT_TOKEN]
        self.eot_id = coconut_token_ids[EOT_TOKEN]

    # ---- public API expected by Trainer ----

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        device = input_ids.device
        batch_size = input_ids.shape[0]

        losses = []
        last_outputs = None

        for i in range(batch_size):
            # strip padding (right-padded)
            seq_len = int(attention_mask[i].sum().item())
            ids_i = input_ids[i, :seq_len].unsqueeze(0)
            mask_i = attention_mask[i, :seq_len].unsqueeze(0)
            lab_i = labels[i, :seq_len].unsqueeze(0)

            loss_i, out_i = self._coconut_forward(model, ids_i, mask_i, lab_i, device)
            losses.append(loss_i)
            last_outputs = out_i

        avg_loss = torch.stack(losses).mean()
        return (avg_loss, last_outputs) if return_outputs else avg_loss

    # ---- internal ----

    def _coconut_forward(self, model, ids, mask, labels, device):
        """Single-sample Coconut forward pass.

        Replaces ``<thought>`` placeholder embeddings with hidden states
        generated autoregressively, then computes the language-modelling loss
        on the answer tokens only.
        """
        # Locate markers
        bot_pos = (ids[0] == self.bot_id).nonzero(as_tuple=True)[0]
        eot_pos = (ids[0] == self.eot_id).nonzero(as_tuple=True)[0]

        # Fallback: if markers are missing, do a standard forward pass
        if len(bot_pos) == 0 or len(eot_pos) == 0:
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            return outputs.loss, outputs

        bot_idx = bot_pos[0].item()
        eot_idx = eot_pos[0].item()
        num_thoughts = eot_idx - bot_idx - 1

        # Mask labels: only keep loss on tokens *after* <eot>
        labels = labels.clone()
        labels[:, : eot_idx + 1] = -100

        # Embed the full sequence
        embed_fn = model.get_input_embeddings()
        input_embs = embed_fn(ids)

        prefix = input_embs[:, : bot_idx + 1]   # [...tokens..., <bot>]
        suffix = input_embs[:, eot_idx:]         # [<eot>, \n, answer...]

        # Fast path: no thought tokens -> standard forward with adjusted labels
        if num_thoughts == 0:
            new_embs = torch.cat([prefix, suffix], dim=1)
            new_mask = torch.ones(1, new_embs.shape[1], device=device)
            outputs = model(
                inputs_embeds=new_embs, attention_mask=new_mask, labels=labels,
            )
            return outputs.loss, outputs

        # ---- Autoregressive continuous thought generation ----

        # First thought: forward on [prefix...<bot>], take last hidden state
        prefix_mask = torch.ones(1, prefix.shape[1], device=device)
        out = model(
            inputs_embeds=prefix,
            attention_mask=prefix_mask,
            output_hidden_states=True,
        )
        thoughts = [out.hidden_states[-1][:, -1:]]

        # Remaining thoughts
        for _ in range(num_thoughts - 1):
            cur_embs = torch.cat([prefix] + thoughts, dim=1)
            cur_mask = torch.ones(1, cur_embs.shape[1], device=device)
            out = model(
                inputs_embeds=cur_embs,
                attention_mask=cur_mask,
                output_hidden_states=True,
            )
            thoughts.append(out.hidden_states[-1][:, -1:])

        # ---- Final forward with continuous thoughts replacing placeholders ----
        new_embs = torch.cat([prefix] + thoughts + [suffix], dim=1)
        new_mask = torch.ones(1, new_embs.shape[1], device=device)

        outputs = model(
            inputs_embeds=new_embs,
            attention_mask=new_mask,
            labels=labels,
        )
        return outputs.loss, outputs
