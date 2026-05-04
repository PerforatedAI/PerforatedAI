from monai.networks.nets import UNet

def get_unet_small(out_channels=4):
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=out_channels,
        channels = (24, 40, 80, 160),
        strides=(2, 2, 2),
        num_res_units=2,
    )

