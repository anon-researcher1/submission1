import sys

def select_model(arg_obj):
    model_type = arg_obj.model_type.lower()
    tk = int(arg_obj.tk)
    channels = arg_obj.channels
    dropout = float(arg_obj.dropout)
    padding_mode = arg_obj.padding_mode

    num_channels = len(channels)
    input_channels = 3

    if model_type == 'rpnet':
        from models.RPNet import RPNet
        model = RPNet(input_channels=input_channels, drop_p=dropout, t_kern=tk, padding_mode=padding_mode)
    elif model_type == 'physnet':
        from models.PhysNet import PhysNet
        model = PhysNet(input_channels=input_channels, drop_p=dropout, t_kern=tk, padding_mode=padding_mode)
    else:
        print('Could not find model specified.')
        sys.exit(-1)

    return model
