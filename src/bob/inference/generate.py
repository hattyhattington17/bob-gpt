import torch

from bob.inference.sampler import greedy
from bob.model.transformer import Bob


def generate(
    model: Bob,
    token_ids: list[int],
    max_new_tokens: int,
    max_seq_len: int,
) -> list[int]:
    """
    autoregressively generates tokens

    
    Args:
        model: the Bob model in eval mode (training behavior like dropout is disabled)
        token_ids: list of token ids (shape T)
        max_new_tokens: max number of tokens to generate
        max_seq_len: maximum context window length, truncates input if exceeded
    Returns:
        full sequence including prompt as a list of token ids
    """
    # x is a tensor of shape (1, T) containing the input token ids, we add a batch dimension of 1 since the model expects batches
    x = torch.tensor([token_ids], dtype=torch.long).to(next(model.parameters()).device)

    for _ in range(max_new_tokens):
        # take last max_seq_len tokens from first batch as input to the model
        x = x[:, -max_seq_len:]
        # run the model to get logits for all positions in the sequence
        logits = model(x)
        # use the logits at the last position to sample the next token id
        next_token = greedy(logits[0, -1, :])
        # add the next token to the sequence and continue
        x = torch.cat([x, torch.tensor([[next_token]], device=x.device)], dim=1)

    return x[0].tolist()
