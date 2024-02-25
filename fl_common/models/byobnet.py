from timm import create_model


def get_byobnet_model(byobnet_type, num_classes):
    if byobnet_type == 'gernet_l':
        byobnet_model = create_model('gernet_l', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gernet_m':
        byobnet_model = create_model('gernet_m', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gernet_s':
        byobnet_model = create_model('gernet_s', pretrained=True, num_classes=num_classes)

