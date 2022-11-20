import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default  = 'cifar')

    # Neural Network setting
    parser.add_argument('-cout', type=int, default  = 12)
    parser.add_argument('-cfeat', type=int, default  = 256)

    # The relay channel
    parser.add_argument('-is_coop', default = True)
    parser.add_argument('-relay_mode', default  = 'PF')
    parser.add_argument('-channel_mode', default = 'awgn')

    parser.add_argument('-sr_link',  default  = 16.0)
    parser.add_argument('-sd_link',  default  = 6.0)
    parser.add_argument('-rd_link',  default  = 6.0)
    parser.add_argument('-sr_rng',  default  = 0)
    parser.add_argument('-sd_rng',  default  = 4.0)
    parser.add_argument('-rd_rng',  default  = 4.0)


    parser.add_argument('-adapt', default  = True)

    # training setting
    parser.add_argument('-epoch', type=int, default  = 400)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 12)
    parser.add_argument('-train_batch_size', type=int, default  = 32)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    args = parser.parse_args()

    return args
