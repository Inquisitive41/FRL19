import numpy as np
import torch
import torch.nn as nn
from torch.fft import rfft
from sympy import nextprime
import pywt

class FRL19Dv2_1(nn.Module):
    def __init__(self, input_dim, output_dim=1, eta=0.01, lambda_reg=0.01, D_max=97):
        super(FRL19Dv2_1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.D = min(nextprime(int(np.sqrt(input_dim))), D_max) if input_dim > 1000 else input_dim // 10 + 1
        self.weights = nn.Parameter(torch.randn(self.D))
        self.eta = eta
        self.lambda_reg = lambda_reg

    def compute_fractal_dim(self, x):
        # Вейвлет-анализ для фрактальной размерности
        coeffs, _ = pywt.cwt(x.cpu().numpy(), scales=np.arange(1, 10), wavelet='morl')
        energy = np.log(np.sum(coeffs**2, axis=-1))
        scales = np.log(np.arange(1, 10))
        D_f = np.polyfit(scales, energy, 1)[0]
        return torch.tensor(D_f, device=x.device)

    def forward(self, x):
        # Реальное FFT и log-scale нормализация
        x_freq = torch.abs(rfft(x))  # Реальная часть FFT
        norm = torch.log(1 + x_freq) / torch.log(1 + x_freq.max(dim=-1, keepdim=True).values) % self.D
        x_D = x_freq * norm

        # Резонанс
        freqs = torch.arange(1, self.D + 1, device=x.device).float() / self.D
        resonance = torch.sin(2 * np.pi * freqs * x_D)
        output = torch.einsum('bnd,d->bn', resonance, self.weights).unsqueeze(-1)
        return output

    def compute_loss(self, pred, y):
        epsilon = 1e-6
        return 1 - (torch.norm(pred - y) ** 2) / (torch.norm(y) ** 2 + epsilon)

    def train_step(self, x, y, t):
        pred = self.forward(x)
        loss = self.compute_loss(pred, y)
        loss.backward()

        with torch.no_grad():
            freqs = torch.arange(1, self.D + 1, device=x.device).float() / self.D
            self.weights -= self.eta * self.weights.grad * torch.cos(2 * np.pi * freqs * t)
            self.weights -= self.eta * self.lambda_reg * self.weights
            self.weights.grad.zero_()

        return loss.item()

def train_frl19dv2_1(X, Y, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRL19Dv2_1(input_dim=X.shape[1]).to(device)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        loss = model.train_step(X, Y, epoch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, D: {model.D}")

    return model

# Тест на зашумлённых данных
X = np.random.randn(1000, 50) + np.random.normal(0, 0.5, (1000, 50))  # Шум SNR=5
Y = (X[:, 0] > 0).astype(np.float32).reshape(-1, 1)
model = train_frl19dv2_1(X, Y)