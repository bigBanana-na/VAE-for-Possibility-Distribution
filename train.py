import torch
import torch.nn as nn
import torch.optim as optim

class VAEPd(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEPd, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = h.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var

def elbo_loss(recon_x, x, mean, log_var):
    # Reconstruction Loss
    recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10))
    # KL Loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_loss

# Data Example
p = torch.tensor([0.5, 0.25, 0.125, 0.125])
q = torch.tensor([0.25, 0.25, 0.25, 0.25])

# VAE for prosibility distribution
vae = VAEPd(input_dim=4, hidden_dim=10, latent_dim=2)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    recon_x, mean, log_var = vae(q)
    loss = elbo_loss(recon_x, p, mean, log_var)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')