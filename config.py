import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--train", type=str, default="yeast", help="the training dataset")
    parser.add_argument("-o", "--output", type=str, default="./output/", help="the path of output model")
    parser.add_argument("-cn", "--create_negative", type=bool, default=False,
                        help="whether to create negative samples based on different conditions")
    parser.add_argument("-b", "--balanced", type=bool, default=False, help="using the balanced atom number")
    parser.add_argument("-i", "--iteration", type=int, default=10, help="the number of running model")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="the max number of epoch")
    parser.add_argument("-ar", "--atom_ratio", type=float, default=0.5,
                        help="the ratio of replaced atoms for negative reaction")
    parser.add_argument("-nr", "--negative_ratio", type=int, default=1,
                        help="the ratio of negative and positive samples")
    parser.add_argument("-s", "--seed", type=int, default=2, help="random seed")

    parser.add_argument('--biochemical_in_dim', type=int, default=2304, 
                       help="the input size of biochemical features")
    parser.add_argument('--biochemical_out_dim', type=int, default=512, 
                       help="the output size of biochemical features")
    parser.add_argument('--dgnn_hid_dim', type=int, default=256, 
                       help="the hidden size of DGN")
    parser.add_argument('--num_dgnn', type=int, default=2, 
                       help="the layer size of DGN")

    parser.add_argument('--use_similarity', type=bool, default=True,
                       help="whether to use similarity features")

    parser.add_argument('--emb_dim', type=int, default=64, help="size of each input sample")
    parser.add_argument('--conv_dim', type=int, default=128, help="the output size of HypergraphConv")
    parser.add_argument('--head', type=int, default=6, help="the head size of HypergraphConv")
    parser.add_argument('--L', type=int, default=2, help="the layer size of HypergraphConv")
    parser.add_argument('--p', type=float, default=0.1, help="dropout probability")
    parser.add_argument('--g_lambda', type=float, default=1, help='the parameter for Gaussian kernel')
    parser.add_argument('--lr', type=float, default=1e-3,  help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()
