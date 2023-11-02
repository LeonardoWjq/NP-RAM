import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, ins_dim: int = 1024, obs_dim: int = 55, out_dim: int = 8):
        super().__init__()

        # instruction compression head
        self.instruction_fc = nn.Sequential(
            nn.Linear(ins_dim, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish()
        )

        # Combine the compressed instruction and observation inputs
        self.combine_fc = nn.Sequential(
            nn.Linear(64 + obs_dim, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.Mish()
        )

        # regression layers to predict mean and variance
        self.mean_fc = nn.Linear(32, out_dim)
        self.sigma_fc = nn.Linear(32, out_dim)

    def forward(self, ins: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        ins = self.instruction_fc(ins)

        ins_and_obs = torch.cat((ins, obs), dim=1)

        feature = self.combine_fc(ins_and_obs)

        # apply mean and variance prediction layers
        mean = self.mean_fc(feature)
        sigma = self.sigma_fc(feature)

        return mean, sigma

if __name__ == '__main__':
    mlp = MLP().to('cuda')
    ins = torch.randn(4, 1024).to('cuda')
    obs = torch.randn(4, 55).to('cuda')
    mean, sigma = mlp(ins, obs)
    print(mean.shape, sigma.shape)