import torch.nn as nn


class SFS_Cnn(nn.Module):
    def __init__(self, in_channel=1, classes=2, kernel_size=3):
        super().__init__()
        # noinspection PyTypeChecker
        output_channel_size = 4
        img_size = 128
        num_of_classes = 2
        final_size_pooling = 64  ## what ever the input gets output to be 64*64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, output_channel_size, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(output_channel_size),
            nn.MaxPool2d(2, stride=None),
            nn.Conv2d(output_channel_size, output_channel_size * 2, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(output_channel_size * 2),
            nn.MaxPool2d(2, stride=None),
            nn.Conv2d(output_channel_size * 2, output_channel_size * 2 * 2, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(output_channel_size * 2 * 2),
            nn.MaxPool2d(2, stride=None),
            nn.AdaptiveAvgPool2d((final_size_pooling, num_of_classes))
        )
        encoder_output_size = output_channel_size * 2 * 2 * final_size_pooling * final_size_pooling

        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(encoder_output_size, encoder_output_size / 2),
            nn.Linear(encoder_output_size / 2, out_features=60))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


net = SFS_Cnn()
print(net)
