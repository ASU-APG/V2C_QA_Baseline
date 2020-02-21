import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument(
        '--info_json',
        type=str,
        default='data/v2c_info.json',
        help='path to the json file containing additional info and vocab')

    parser.add_argument(
        '--cap_info_json',
        type=str,
        default='data/msrvtt_new_info.json',
        help='path to the json file containing additional info and vocab')

    parser.add_argument(
        '--caption_json',
        type=str,
        default='data/V2C_MSR-VTT_caption.json',
        help='path to the processed video caption json')

    parser.add_argument(
        '--v2c_qa_json',
        type=str,
        default='data/v2cqa_v1_train.json',
        help='path to the json file containing V2C-QA task.')

    parser.add_argument(
        '--feats_dir',
        nargs='*',
        type=str,
        default='data/feats/resnet152/',
        help='path to the directory containing the preprocessed fc feats')

    # Model settings
    parser.add_argument(
        "--cap_max_len",
        type=int,
        default=28,
        help='max length of captions(containing <sos>, <eos>)')

    parser.add_argument(
        "--int_max_len",
        type=int,
        default=21,
        help='max length of captions(containing <sos>, <eos>)')

    parser.add_argument(
        "--eff_max_len",
        type=int,
        default=26,
        help='max length of captions(containing <sos>, <eos>)')

    parser.add_argument(
        "--att_max_len",
        type=int,
        default=8,
        help='max length of captions(containing <sos>, <eos>)')

    parser.add_argument(
        '--num_layers', type=int, default=1, help='number of layers in the Transformers')

    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.2,
        help='strength of dropout in the Language Model RNN')

    parser.add_argument(
        '--dim_word',
        type=int,
        default=512,
        help='the encoding size of each token in the vocabulary, and the video.')

    parser.add_argument(
        '--dim_model',
        type=int,
        default=512,
        help='size of the rnn hidden layer')

    parser.add_argument(
        '--dim_vis_feat',
        type=int,
        default=2048,
        help='dim of features of video frames')

    # 12-12 8 6
    parser.add_argument(
        '--num_head',
        type=int,
        default=8,
        help='Numbers of head in transformers.')

    parser.add_argument(
        '--num_layer',
        type=int,
        default=6,
        help='Numbers of layers in transformers.')

    parser.add_argument(
        '--dim_head',
        type=int,
        default=64,
        help='Dimension of the attention head.')

    parser.add_argument(
        '--dim_inner',
        type=int,
        default=1024,
        help='Dimension of inner feature in Encoder/Decoder.')

    # Optimization: General
    parser.add_argument(
        '--epochs', type=int, default=600, help='number of epochs')

    parser.add_argument(
        '--warm_up_steps', type=int, default=5000, help='Warm up steps.')

    parser.add_argument(
        '--batch_size', type=int, default=48, help='minibatch size')

    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        default=30,
        help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='save',
        help='directory to store check pointed models')

    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default='./save/model_cap-int.pth',
        help='directory to load check pointed models')

    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()

    return args
