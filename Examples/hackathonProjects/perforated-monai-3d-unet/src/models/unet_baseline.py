from monai.networks.nets import UNet

def get_unet(out_channels=4):
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=out_channels,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",  # optional but recommended
    )
