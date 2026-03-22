import torch

from bob.config import ModelConfig
class Bob(torch.nn.Module):
    """ GPT Language Model """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # placeholder for RMS norm
        self.norm = torch.nn.Identity()

        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        if  (config.tie_embeddings):
            self.lm_head.weight = self.embeddings.weight 
 
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Generate logits for the next token 
        input - tensor of token ids - shape (B, T) where B is batch size and T is sequence length
        output - tensor of logits - shape (B, T, vocab_size)
        """
        print("batch size: B =", input.shape[0])  
        print("sequence length: T =", input.shape[1])  

        # embeddings - shape (B, T, d_model)
        # use the embedding table shape vocab_size x d_model to map token ids to embeddings
        hidden_state = self.embeddings(input)
        print("embeddings shape:", hidden_state.shape)

        # transformer layers
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        # final norm
        hidden_state = self.norm(hidden_state)
        # map hidden state to logits - shape (B, T, vocab_size)
        logits = self.lm_head(hidden_state)
        print("logits shape:", logits.shape)
        return logits

class TransformerBlock(torch.nn.Module):
    """ Transformer block """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity map 
        return x