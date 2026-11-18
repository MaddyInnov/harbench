"""Baseline models: DeepConvLSTM and TinyHAR.

Both take (B, C, T) tensors. Kept deliberately standard -- these are the
reference baselines from the windowed-gaps research, not the novelty.
"""

import torch
import torch.nn as nn


class DeepConvLSTM(nn.Module):
    """Ordóñez & Roggen 2016: 4x conv(64, k=5) -> 2x LSTM(128) -> softmax."""

    def __init__(self, n_channels=9, n_classes=6, conv_filters=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.0):
        super().__init__()
        convs = []
        in_ch = n_channels
        for _ in range(4):
            convs += [nn.Conv1d(in_ch, conv_filters, kernel_size=5, padding=2),
                      nn.ReLU(inplace=True)]
            in_ch = conv_filters
        self.convs = nn.Sequential(*convs)
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        h = self.convs(x)            # (B, 64, T)
        h = h.permute(0, 2, 1)       # (B, T, 64)
        _, (hn, _) = self.lstm(h)    # hn: (layers, B, 128)
        return self.classifier(hn[-1])


class TinyHAR(nn.Module):
    """Compact conv + transformer baseline in the spirit of Zhou et al. 2022.

    Per-channel temporal convs -> cross-channel fusion conv ->
    small transformer over time -> mean pool -> classifier.
    """

    def __init__(self, n_channels=9, n_classes=6, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.channel_conv = nn.Conv1d(n_channels, n_channels * 4, kernel_size=5,
                                      padding=2, groups=n_channels)
        self.fusion = nn.Sequential(
            nn.Conv1d(n_channels * 4, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        h = torch.relu(self.channel_conv(x))
        h = self.fusion(h)              # (B, d_model, T)
        h = h.permute(0, 2, 1)          # (B, T, d_model)
        h = self.transformer(h)
        h = h.mean(dim=1)               # mean pool over time
        return self.classifier(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SEBlock(nn.Module):
    """Squeeze-and-excitation over channels."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.fc(x.mean(dim=2))
        return x * w.unsqueeze(-1)


class InceptionHAR(nn.Module):
    """Multi-scale 1D-conv encoder (k=1,3,5,11) + SE attention + MLP head.

    The encoder mirrors the InceptionKAN paper's front end; the KAN head
    is replaced by a plain MLP here to isolate the encoder's contribution.
    """

    def __init__(self, n_channels=6, n_classes=12, branch_ch=32, dropout=0.2,
                 aux_purity=False):
        super().__init__()
        self.aux_purity = aux_purity
        self.branches = nn.ModuleList([
            nn.Conv1d(n_channels, branch_ch, kernel_size=1),
            nn.Conv1d(n_channels, branch_ch, kernel_size=3, padding=1),
            nn.Conv1d(n_channels, branch_ch, kernel_size=5, padding=2),
            nn.Conv1d(n_channels, branch_ch, kernel_size=11, padding=5),
        ])
        self.se = SEBlock(branch_ch * 4)
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(branch_ch * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, n_classes)
        if aux_purity:
            self.purity_head = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, C, T)
        h = torch.cat([b(x) for b in self.branches], dim=1)
        h = self.se(h)
        h = self.head(h.mean(dim=2))
        logits = self.classifier(h)
        if self.aux_purity:
            return logits, torch.sigmoid(self.purity_head(h)).squeeze(-1)
        return logits


class TinyHAR_MT(TinyHAR):
    """TinyHAR with an auxiliary window-purity regression head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.classifier = Sequential(Dropout, Linear(d_model, n_classes))
        d_model = self.classifier[1].in_features
        self.purity_head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = torch.relu(self.channel_conv(x))
        h = self.fusion(h)
        h = h.permute(0, 2, 1)
        h = self.transformer(h)
        h = h.mean(dim=1)
        logits = self.classifier(h)
        purity = torch.sigmoid(self.purity_head(h)).squeeze(-1)
        return logits, purity


class TinyHARAbstain(TinyHAR):
    """TinyHAR with an extra abstain output (K+1 logits) for selective
    classification via the Deep Gamblers loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.classifier[1].in_features
        n_classes = self.classifier[1].out_features
        self.classifier = nn.Sequential(
            nn.Dropout(self.classifier[0].p),
            nn.Linear(d_model, n_classes + 1),
        )


class TinyHARWide(TinyHAR):
    """Higher-capacity TinyHAR: wider transformer, extra layer."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("d_model", 128)
        kwargs.setdefault("n_heads", 8)
        kwargs.setdefault("n_layers", 3)
        super().__init__(*args, **kwargs)


def build_model(name, n_channels, n_classes, seed=42):
    torch.manual_seed(seed)
    if name == "deepconvlstm":
        return DeepConvLSTM(n_channels=n_channels, n_classes=n_classes)
    if name == "tinyhar":
        return TinyHAR(n_channels=n_channels, n_classes=n_classes)
    if name == "inception":
        return InceptionHAR(n_channels=n_channels, n_classes=n_classes)
    if name == "tinyhar-mt":
        return TinyHAR_MT(n_channels=n_channels, n_classes=n_classes)
    if name == "inception-mt":
        return InceptionHAR(n_channels=n_channels, n_classes=n_classes,
                            aux_purity=True)
    if name == "tinyhar-abstain":
        return TinyHARAbstain(n_channels=n_channels, n_classes=n_classes)
    if name == "tinyhar-wide":
        return TinyHARWide(n_channels=n_channels, n_classes=n_classes)
    if name in ("tinyhar-grkan", "gated-deltanet", "rwkv-ts", "moment", "inceptionkan"):
        from .models_new import build_new_model
        return build_new_model(name, n_channels, n_classes, seed=seed)
    raise ValueError(f"Unknown model: {name}")
