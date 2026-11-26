import torch
import torch.nn as nn

#Section Architecture du réseau de neurones

class UnetCNN(nn.Module) :
    def __init__(self, num_classes = 55):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32 #16
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32 #16
                           ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32 #16
                      , out_channels=32 #16
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32 #16
                           ),
            nn.ReLU(inplace=True)  
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=32 #16
                      , out_channels=64 #32
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64 #32
                           ),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=64 #32
                      , out_channels=64 #32
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64 #32
                           ),
            nn.ReLU(inplace=True),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=64 #32
                      , out_channels=128 #64
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128 #64
                           ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128 #64
                      , out_channels=128 #64
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128 #64
                           ),
            nn.ReLU(inplace=True),

        )

        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=128 #64
                      , out_channels=256 #128
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256 #128
                           ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256 #128
                      , out_channels=256 #128
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256 #128
                           ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=384 #192
                      , out_channels = 128 #64
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128 #64
                           ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128 #64
                      , out_channels=128 #64
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128 #64
                           ),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels=192 #96
                      , out_channels= 64 #32
                      , kernel_size = 3, padding=1, stride=1),
            nn.BatchNorm2d(64 #32
                           ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64 #32
                      , out_channels=64 #32
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64 #32
                           ),
            nn.ReLU(inplace=True),

        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels=96 #48
                      , out_channels=32 #16
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32 #16
                           ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32 #16
                      , out_channels=32 #16
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32 #16
                           ),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.finalconv = nn.Conv2d(in_channels = 32 #16
                                   , out_channels=num_classes, kernel_size=1, stride=1)


    def forward(self, x):
        x = self.down1(x)
        x_concatdown1 = x
        x = self.pool(x)

        x = self.down2(x)
        x_concatdown2 = x
        x = self.pool(x)

        x = self.down3(x)
        x_concatdown3 = x
        x = self.pool(x)

        x = self.down4(x)
        x = self.upsample(x)
        x = torch.cat([x,x_concatdown3], dim=1)

        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x_concatdown2], dim=1)

        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x_concatdown1], dim=1)

        x = self.up3(x)
        x = self.finalconv(x)
        return x
