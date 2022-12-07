import numpy as np
import torch
from scipy.sparse import coo_array

# lrp-tutorial sub repo
from lrp_tutorial import utils as tut_utils

from util.util_gamma_rule import \
    conv_matrix_from_pytorch_layer, \
    global_conv_matrix_from_pytorch_layer, \
    back_matrix

def back_matrix_recovery_test():
    """
    Checks if the back matrix recovers original activations,
    when gamma=0.
    """
    gamma=0

    for mat in np.random.normal(0,1, (10, 3, 2)):
        mat /= mat.sum(axis=0, keepdims=True)
        
        for a in np.random.normal(0,1,(10,2)):
            back = back_matrix(mat, a, gamma)
            assert np.allclose(back @ mat @ a, a), f"{back @ mat @ a}, {a}"


def equality_test__conv_layer__conv_matrix():
    inp_feats, out_feats = 3, 6
    stride=1

    seed = 0

    for inp_img_shape in [(5, 5), (10, 10), (5, 7)]:

        for kernel_size in ((2,2), (3,3), (2,3)):
            for padding in (0,1,2,3):
                seed += 1
                torch.random.manual_seed(seed)

                conv_layer = torch.nn.Conv2d(in_channels=inp_feats, out_channels=out_feats, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

                inp_img = torch.normal(0, 1, size=(1, inp_feats, *inp_img_shape)).abs()
                out_img = conv_layer.forward(inp_img)
                out_img_shape = out_img.shape[2:]

                # indirect computation of the global transiton matrix
                trans_blocks = [[
                            conv_matrix_from_pytorch_layer(conv_layer, inp_img_shape, inp_feat_no, out_feat_no).toarray()
                        for inp_feat_no in range(inp_feats)]
                    for out_feat_no in range(out_feats)]
                trans_indirect = coo_array(np.block(trans_blocks))

                # direct computation of the global transiton matrix
                trans = global_conv_matrix_from_pytorch_layer(conv_layer, (inp_feats, *inp_img_shape), (out_feats, *out_img_shape))

                # check equality of direct or indirect method
                assert np.all((trans_indirect != trans).data), "Inequality of direct or indirect method."

                # check equality of results between matrix representation and conv layer.
                out_vec = trans @ inp_img.flatten()
                a, b = out_img.detach(), out_vec.reshape(out_img.shape)
                assert np.allclose(a, b, atol=1e-4), f"Output result by global transition matrix is unequal to direct pytorch computation. Max difference: {(a-b).abs().max()}"

def equality_test__forward_hook__lrp_back_matrix():
    inp_feats, out_feats = 3, 6
    stride=1

    seed = 0

    for inp_img_shape in [(5, 5), (10, 10), (5, 7)]:

        # for kernel_size in ((2,2), (3,3), (2,3)):
        #     for padding in (0,1,2,3):
        for kernel_size in ((3,3), ):
            for padding in (1, ):
                seed += 1
                torch.random.manual_seed(seed)

                conv_layer = torch.nn.Conv2d(in_channels=inp_feats, out_channels=out_feats, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

                # create random input and compute output
                inp_img = torch.normal(0, 1, size=(1, inp_feats, *inp_img_shape)).abs()
                out_img = conv_layer.forward(inp_img)
                out_img = torch.nn.functional.relu(out_img)
                out_img_shape = out_img.shape[2:]

                # create something that could reasonably be the later layers relevancies score.
                out_rel = torch.normal(0, 1, size=out_img.shape) * out_img.detach()

                for gamma in [0, .1, .25, 1, 1000000]:
                    inp_img = (inp_img.data).requires_grad_(True)

                    # computation of input relevancies by forward hook
                    rho = lambda p: p + gamma*p.clamp(min=0)
                    z = tut_utils.newlayer(conv_layer, rho).forward(inp_img)         # step 1
                    s = (out_rel/z).data                                             # step 2
                    (z*s).sum().backward(); c = inp_img.grad                         # step 3
                    inp_rel__forward_hook = (inp_img*c).data 
                    
                    for delete_unnecessary_rows in [False, True]:
                        # computation of input relevancies by LRP backward matrix
                        forw = global_conv_matrix_from_pytorch_layer(conv_layer, (inp_feats, *inp_img_shape), (out_feats, *out_img_shape))
                        back = back_matrix(forw, inp_img.data.flatten(), gamma=gamma, delete_unnecessary_rows=delete_unnecessary_rows)
                        inp_rel__lrp_matrix = back @ out_rel.flatten()

                        # check equality of methods
                        a, b = inp_rel__forward_hook.detach(), inp_rel__lrp_matrix.reshape(inp_rel__forward_hook.shape)
                        err = ((a-b).abs() / np.minimum(a, b)).abs().max() # calculate relative error
                        assert err < 1e-3, f"Unequal results. Max relative error: {err}"

if __name__ == "__main__":
    # back_matrix_recovery_test()
    equality_test__conv_layer__conv_matrix()
    equality_test__forward_hook__lrp_back_matrix()