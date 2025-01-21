import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device='cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
	raise Exception('Please switch to GPU mode or comment out this line')

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # Output: N x features_d x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # Output: N x features_d*2 x 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # Output: N x features_d*4 x 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # Output: N x features_d*8 x 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # Output: N x 1 x 1 x 1
            nn.Sigmoid(),  # Output: N x 1 x 1 x 1 (probability)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # Output: N x features_g*16 x 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # Output: N x features_g*8 x 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # Output: N x features_g*4 x 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # Output: N x features_g*2 x 32 x 32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),  # Output: N x channels_img x 64 x 64
            nn.Tanh()  # Output: N x channels_img x 64 x 64 (pixel values normalized)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
	N, in_channels, H, W = 8, 3, 64, 64
	noise_dim = 100
	x = torch.randn((N, in_channels, H, W))
	disc = Discriminator(in_channels, 8)
	initialize_weights(disc)
	assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
	gen = Generator(noise_dim, in_channels, 8)
	initialize_weights(gen)
	z = torch.randn((N, noise_dim, 1, 1))
	assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

if __name__ == "__main__":
    test()

