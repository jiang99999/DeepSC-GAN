
import argparse


def para_config():
    parser = argparse.ArgumentParser()

    # preprocessing parameters
    parser.add_argument('--input-data-dir', default='txt/en', type=str)
    parser.add_argument('--output-train-dir', default='txt/train_data.pkl', type=str)
    parser.add_argument('--output-test-dir', default='txt/test_data.pkl', type=str)
    parser.add_argument('--output-vocab', default='txt/vocab.json', type=str)
    parser.add_argument('--log-save-path', default='F:/jupyter/semantic/star-DeepSC/log', type=str)

    parser.add_argument('--train-save-path', default='F:/jupyter/semantic/star-DeepSC/data/txt/train_data.pkl', type=str)
    parser.add_argument('--test-save-path', default='F:/jupyter/semantic/star-DeepSC/data/txt/test_data.pkl', type=str)
    parser.add_argument('--vocab-path', default='F:/jupyter/semantic/star-DeepSC/data/txt/vocab.json', type=str)

    # Training parameters
    parser.add_argument('--bs', default=64, type=int, help='The training batch size')
    parser.add_argument('--shuffle-size', default=22234, type=int, help='The training shuffle size')
    parser.add_argument('--lr', default=5e-4, type=float, help='The training learning rate')
    parser.add_argument('--epochs', default=60, type=int, help='The training number of epochs')
    parser.add_argument('--train-with-mine',  action='store_true',
                    help='If added, the network will be trained WITH Mutual Information')
    parser.add_argument('--checkpoint-path', default='F:/jupyter/semantic/star-DeepSC/checkpoint', type=str,
                        help='The path to save model')
    parser.add_argument('--Bob-checkpoint-path', default='F:/jupyter/semantic/Attack-DeepSC/checkpoint/Bob', type=str,
                        help='The path to save Bob model')
    parser.add_argument('--Eve-checkpoint-path', default='F:/jupyter/semantic/Attack-DeepSC/checkpoint/Eve', type=str,
                        help='The path to save Eve model')
    parser.add_argument('--Encry-checkpoint-path', default='F:/jupyter/semantic/Attack-DeepSC/checkpoint/Encry', type=str,
                        help='The path to save Encryptor model')
    parser.add_argument('--Decry-checkpoint-path', default='F:/jupyter/semantic/Attack-DeepSC/checkpoint/Decry', type=str,
                        help='The path to save Decryptor model')
    parser.add_argument('--max-length', default=30, type=int, help='The path to save model')
    parser.add_argument('--channel', default='AWGN', type=str, help='Choose the channel to simulate')

    # Model parameters
    parser.add_argument('--encoder-num-layer', default=4, type=int, help='The number of encoder layers')
    parser.add_argument('--encoder-d-model', default=128, type=int, help='The output dimension of attention')
    parser.add_argument('--encoder-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--encoder-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--encoder-dropout', default=0.1, type=float, help='The encoder dropout rate')

    parser.add_argument('--decoder-num-layer', default=4, type=int, help='The number of decoder layers')
    parser.add_argument('--decoder-d-model', default=128, type=int, help='The output dimension of decoder')
    parser.add_argument('--decoder-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--decoder-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--decoder-dropout', default=0.1, type=float, help='The decoder dropout rate')

    #star-transformer
    parser.add_argument('--cycle-num', default=8, type=int, help='Number of inner cycles')
    parser.add_argument('--cycle-layers', default=8, type=int, help='Number of outer cycles')
    
    
    # Other parameter settings
    parser.add_argument('--train-snr', default=3, type=int, help='The train SNR')
    parser.add_argument('--test-snr', default=6, type=int, help='The test SNR')
    # Mutual Information Model Parameters


    args = parser.parse_known_args()[0]

    return args

