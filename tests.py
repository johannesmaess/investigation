import numpy as np
import torch
from scipy.sparse import coo_array

from util.util_gamma_rule import \
    conv_matrix_from_pytorch_layer, \
    global_conv_matrix_from_pytorch_layer

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


def conv_as_matrix_equality_test():
    inp_feats, out_feats = 2, 3
    stride=1

    for inp_img_shape in ((5, 5), (10, 10), (5, 7)):
        for kernel_size in ((2,2), (3,3), (2,3)):
            for padding in (0,1,2,3):
                torch.random.manual_seed(1)

                conv_layer = torch.nn.Conv2d(in_channels=inp_feats, out_channels=out_feats, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

                inp_img = torch.normal(0, 1, size=(1, inp_feats, *inp_img_shape)).abs()
                out_img = conv_layer.forward(inp_img)
                out_img_shape = out_img.shape[2:]

                # indirect computation of the global transiton matrix
                trans_blocks = [[
                            conv_matrix_from_pytorch_layer(conv_layer, inp_img_shape, out_feat_no, inp_feat_no).toarray()
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

if __name__ == "__main__":
    # back_matrix_recovery_test()
    conv_as_matrix_equality_test()