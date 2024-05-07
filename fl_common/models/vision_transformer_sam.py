from timm import create_model


def get_vision_transformer_sam_model(vision_transformer_sam_type, num_classes):
    """
    Function to get a Vision Transformer SAM (Spatial Attention Module) model of a specified type.

    Parameters:
        vision_transformer_sam_type (str): Type of Vision Transformer SAM model to be used.
                                           Choices include various types such as 'samvit_base_patch16',
                                           'samvit_large_patch16', 'samvit_huge_patch16', etc.
        num_classes (int): Number of classes for the classification task.

    Returns:
        vision_transformer_sam_model: Vision Transformer SAM model instance with specified architecture and number of classes.

    Raises:
        ValueError: If the specified vision_transformer_sam_type is not one of the supported architectures.
    """
    if vision_transformer_sam_type == "samvit_base_patch16":
        try:
            vision_transformer_sam_model = create_model('samvit_base_patch16',
                                                        pretrained=True,
                                                        num_classes=num_classes)
        except ValueError:
            vision_transformer_sam_model = create_model('samvit_base_patch16',
                                                        pretrained=False,
                                                        num_classes=num_classes)
    elif vision_transformer_sam_type == "samvit_large_patch16":
        try:
            vision_transformer_sam_model = create_model('samvit_large_patch16',
                                                        pretrained=True,
                                                        num_classes=num_classes)
        except ValueError:
            vision_transformer_sam_model = create_model('samvit_large_patch16',
                                                        pretrained=False,
                                                        num_classes=num_classes)
    elif vision_transformer_sam_type == "samvit_huge_patch16":
        try:
            vision_transformer_sam_model = create_model('samvit_huge_patch16',
                                                        pretrained=True,
                                                        num_classes=num_classes)
        except ValueError:
            vision_transformer_sam_model = create_model('samvit_huge_patch16',
                                                        pretrained=False,
                                                        num_classes=num_classes)
    elif vision_transformer_sam_type == "samvit_base_patch16_224":
        try:
            vision_transformer_sam_model = create_model('samvit_base_patch16_224',
                                                        pretrained=True,
                                                        num_classes=num_classes)
        except ValueError:
            vision_transformer_sam_model = create_model('samvit_base_patch16_224',
                                                        pretrained=False,
                                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Vision Transformer Sam Architecture: {vision_transformer_sam_type}')

    return vision_transformer_sam_model
