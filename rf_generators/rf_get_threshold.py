import argparse
import sys
sys.path.append("../")
import rf_collection as rfc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',
                        type=str,
                        default='rf',
                        choices=['rf', 'quantile', 'cochran'],
                        help='criteria for deriving threshold')
    parser.add_argument('--rf_type',
                        type=str,
                        default='gauss',
                        choices=['gauss', 'chisq'],
                        help='type of RF')
    parser.add_argument('--dim',
                        type=int,
                        default=None,
                        help='dimension of RF')
    parser.add_argument('--L',
                        type=int,
                        default=32,
                        nargs='+',
                        help='dimensions of RF, can be a list')
    parser.add_argument('--scale', type=float, default=10, help='scale of RF')
    parser.add_argument('--df', type=int, default=1, help='df for chisq rf')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='confidence level. In case test is quantile, then 1-alpha is used'
    )
    args = parser.parse_args()

    thresh = rfc.get_threshold(
        dim=args.dim,
        test=args.test,
        rf_type=args.rf_type,  #gauss chisq
        alpha=args.alpha,
        scale=args.scale,
        df=args.df,
        L=args.L)
    print(thresh)
