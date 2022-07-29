def get_datapath(args):
    #-----------------------------------------------------------------------
    # Data definition
    #-----------------------------------------------------------------------
    # Need to clean up this block later with path to files
    df = args.df
    data_path = {'source': {}, 'target': {}}

    ## Uncertainty within the model

    if args.data == 'synthetic':
        # Target data - Gaussian RF
        data_path['target'][
            'data'] = '../random_fields/EC/generated_rf/32_10000/generated_rf_1_scale_10/gauss/'
        data_path['target'][
            'mask'] = '../random_fields//EC/generated_rf/32_10000/generated_rf_1_scale_10/gauss_mask/'

        # Source data - different Chisq fields
        data_path['source'][
            'data'] = '../random_fields//EC/generated_rf/32/generated_rf_1/chisq/df%d/' % df
        data_path['source'][
            'mask'] = '../random_fields//EC/generated_rf/32/generated_rf_1/chisq/df%d_mask/' % df
        data_path['source']['base'] = None

    elif 'celeba' in args.data or 'afhq' in args.data:
        size = 64
        # Target data - Gaussian RF
        data_path['target'][
            'data'] = '../random_fields//EC/generated_rf/64_10000/generated_rf_1_scale_10/gauss/'
        data_path['target'][
            'mask'] = '../random_fields//EC/generated_rf/64_10000/generated_rf_1_scale_10/gauss_mask/'

        # Source data - generated uncertainty
        data_path['source'][
            'data'] = '../random_fields//EC/generated_rf/64/%s/within/std.pkl' % args.data
        data_path['source']['mask'] = None
        data_path['source'][
            'base'] = '../random_fields//EC/generated_rf/64/%s/within/mean.pkl' % args.data
        args.img_size = size
        args.pval_targ = 0.99

    return data_path
