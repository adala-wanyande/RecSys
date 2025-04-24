import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT4Rec(nn.Module):
    def __init__(self, 
                 num_items, 
                 hidden_size=128, 
                 max_seq_len=20, 
                 num_layers=2, 
                 num_heads=4, 
                 dropout=0.1):
        """
        num_items: Number of real items (excluding padding and mask)
        """
        super(BERT4Rec, self).__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Vocabulary: 0 = padding, 1 = [MASK], 2...num_items+1 = actual items
        vocab_size = num_items + 2

        # Item and Position Embeddings
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        attention_mask: Tensor of shape (batch_size, seq_len), 1 = keep, 0 = pad
        """
        batch_size, seq_len = input_ids.size()

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        item_embed = self.item_embedding(input_ids)
        pos_embed = self.position_embedding(position_ids)

        x = item_embed + pos_embed
        x = self.layer_norm(self.dropout(x))

        if attention_mask is not None:
            # Invert attention mask for PyTorch Transformer (True = mask)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        logits = self.output_layer(x)

        return logits  # (batch_size, seq_len, vocab_size)
