from timm import create_model


def get_vision_transformer_relpos_model(
        vision_transformer_relpos_type, num_classes):
    """
    Function to get a Vision Transformer Relative Position model of a specified type.

    Parameters:
        vision_transformer_relpos_type (str): Type of Vision Transformer Relative Position model to be used.
        num_classes (int): Number of classes for the classification task.

    Returns:
        vision_transformer_relpos_model: Vision Transformer Relative Position model instance with specified architecture and number of classes.

    Raises:
        ValueError: If the specified vision_transformer_relpos_type is not one of the supported architectures.
    """
    valid_types = {
        'vit_relpos_base_patch32_plus_rpn_256',
        'vit_relpos_base_patch16_plus_240',
        'vit_relpos_small_patch16_224',
        'vit_relpos_medium_patch16_224',
        'vit_relpos_base_patch16_224',
        'vit_srelpos_small_patch16_224',
        'vit_srelpos_medium_patch16_224',
        'vit_relpos_medium_patch16_cls_224',
        'vit_relpos_base_patch16_cls_224',
        'vit_relpos_base_patch16_clsgap_224',
        'vit_relpos_small_patch16_rpn_224',
        'vit_relpos_medium_patch16_rpn_224',
        'vit_relpos_base_patch16_rpn_224'}

    if vision_transformer_relpos_type not in valid_types:
        raise ValueError(
            f'Unknown Vision Transformer Relative Position Architecture: {vision_transformer_relpos_type}')

    try:
        return create_model(
            vision_transformer_relpos_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{vision_transformer_relpos_type} - Error loading pretrained model: {e}")
        return create_model(
            vision_transformer_relpos_type,
            pretrained=False,
            num_classes=num_classes)
