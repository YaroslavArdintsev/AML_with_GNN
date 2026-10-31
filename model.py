import torch
from torch import nn

def mlp(input_size: int, output_size: int):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU(),
        nn.Linear(output_size, output_size),
        nn.ReLU(),
    )

class TransactionGNN(nn.Module):
    def __init__(
            self,
            categoty_sizes,
            numeric_size: int = 7,
            hidden_size: int = 64,
            embedding_size: int = 8,
            depth: int = 2,
            dropout: float | int = 0.1
    ):
        super().__init__()
        self.depth = depth

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_size, padding_idx=0)
            for size in categoty_sizes
        ])

        event_input_size = (
            numeric_size + len(categoty_sizes) + embedding_size
        )

        self.event_encoder = mlp(event_input_size, hidden_size)
        self.node_encoder = mlp(12, hidden_size)

        self.message_layers = nn.ModuleList([
            mlp(2 * hidden_size + 1, hidden_size)
            for _ in range(depth)
        ])

        self.update_layers = nn.ModuleList([
            mlp(3 * hidden_size, hidden_size)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def encode_event(self, numeric, categorical):
        parts = [numeric]
        for column, embedding in enumerate(self.embeddings):
            parts.append(embedding(categorical[..., column]))
        return self.event_encoder(torch.cat(parts, dim=-1))

    def encode_nodes(self, tree, level):
        own = self.node_encoder(tree['X'])

        if level == 0:
            return own

        mask = tree['mask']
        rows, neighbors = mask.shape

        child = self.encode_nodes(tree['child'], level - 1)
        child = child.reshape(rows, neighbors, -1)

        edge = self.encode_event(tree['num'], tree['cat'])

        message = self.message_layers[level - 1](
            torch.cat([child, edge, tree['age']], dim=-1)
        )

        def masked_mean(values, valid):
            weights = valid.unsqueeze(-1).to(values.dtype)
            return (
                (values * weights).sum(dim=1)
                / weights.sum(dim=1).clamp_min(1)
            )

        middle = neighbors // 2
        incoming = masked_mean(
            message[:, :middle],
            mask[:, :middle]
        )
        outgoing = masked_mean(
            message[:, middle:],
            mask[:, middle:]
        )

        return self.update_layers[level - 1](
            torch.cat([own, incoming, outgoing], dim=-1)
        )

    def forward(self, batch):
        event = self.encode_event(batch['num'], batch['cat'])
        nodes = self.encode_nodes(batch['tree'], self.depth)
        nodes = nodes.reshape(event.shape[0], 2, -1)

        sender = nodes[:, 0]
        receiver = nodes[:, 1]

        return self.head(
            torch.cat([sender, receiver, event], dim=-1)
        ).squeeze(-1)