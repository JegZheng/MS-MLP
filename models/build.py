from .ms_mlp import MS_MLP

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'ms_mlp':
        model = MS_MLP(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.MSMLP.PATCH_SIZE,
                                in_chans=config.MODEL.MSMLP.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.MSMLP.EMBED_DIM,
                                depths=config.MODEL.MSMLP.DEPTHS,
                                shift_size=config.MODEL.MSMLP.SHIFT_SIZE,
                                shift_dist=config.MODEL.MSMLP.SHIFT_DIST,
                                mix_size=config.MODEL.MSMLP.MIX_SIZE,
                                mlp_ratio=config.MODEL.MSMLP.MLP_RATIO,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.MSMLP.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

