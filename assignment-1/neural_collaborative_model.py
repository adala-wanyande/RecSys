import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embed_size, hidden_layers):
        super(NeuralCollaborativeFiltering, self).__init__()

        # GMF Embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embed_size)
        self.gmf_item_embedding = nn.Embedding(num_items, embed_size)

        # MLP Embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, embed_size)
        self.mlp_item_embedding = nn.Embedding(num_items, embed_size)

        # MLP Layers
        mlp_layers = []
        input_size = embed_size * 2
        for layer_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp_layers = nn.Sequential(*mlp_layers)

        # Fusion Layer: concatenate GMF and MLP outputs
        self.fusion_layer = nn.Linear(embed_size + hidden_layers[-1], 1)

        # Output Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        # GMF pathway
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item  # element-wise product

        # MLP pathway
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.fusion_layer(combined)
        return self.sigmoid(prediction).squeeze()
