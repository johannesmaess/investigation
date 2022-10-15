import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrow, Circle

from util.util_lrp_transformation_visualization import h0_W_h1_R1_C_R0

def visualize_mc_matrix_transform(M, IN, c="blue", ax=None, patches={}, annotate_with=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
    def rm(label):
        if label in patches: 
            patches[label].remove()
            
    def update_patch(label, patch):
        rm(label)
        patches[label] = ax.add_patch(patch)
        
    ax.set_xlim((-0.05,1.05))
    ax.set_ylim((-0.05,1.05))
    
    cond1 = M[:, 0]
    cond2 = M[:, 1]
    
    # transformed unit vectors:
    joint1 = cond1 * IN[0]
    joint2 = cond2 * IN[1]
    
    marginalized = joint1+joint2
    
    hw = 0.02
    
    update_patch('a1', FancyArrow(0,0, *joint1, head_width=hw, color=c))
    update_patch('a2', FancyArrow(0,0, *joint2, head_width=hw, color=c))
    
    update_patch('a3', FancyArrow(*joint1, *joint2, head_width=hw, color=c))
    update_patch('a4', FancyArrow(*joint2, *joint1, head_width=hw, color=c))
    
    update_patch('p1', Circle(marginalized, radius=.01)) # make this larger
    
    update_patch('a5', FancyArrow(0, 0, *cond1, color=c, alpha=.3, head_width=.0005, linestyle='dotted'))
    update_patch('a6', FancyArrow(0, 0, *cond2, color=c, alpha=.3, head_width=.0005, linestyle='dotted'))

    if annotate_with:
        l1, l2, matrix_name, result_name = annotate_with
        
        def update_annotation(label, annotation, coordinates):
            rm(label)
            patches[label] = ax.annotate(annotation, coordinates+np.array([.03, -.03]))
            
        update_annotation('ann_joint_1',      f"$p({l2}|{l1}_1)p({l1}_1)$", joint1)
        update_annotation('ann_joint_2',      f"$p({l2}|{l1}_2)p({l1}_2)$", joint2)
        
        update_annotation('ann_cond_1',       matrix_name + "$_{:, 2}=" + f"p({l2}|{l1}_1)$", cond1)
        update_annotation('ann_cond_2',       matrix_name + "$_{:, 2}=" + f"p({l2}|{l1}_2)$", cond2)
        
        update_annotation('ann_marginalized', result_name + "$=p("+l2+")$", marginalized)
        
        # annotate axis
        ax.set_xlabel("$p("+l2+"_1)$")
        ax.set_ylabel("$p("+l2+"_2)$")
    
    ax.plot((0,1), (1,0), linestyle='dotted', color="green")
    # ax.plot((0,2), (2,0), linestyle='dashed')
    
    # plt.show()
    return patches
            

# The function to be called anytime a slider's value changes
def update(_):
    # perspective by sliders:
    w11 = w11_slider.val
    w12 = w12_slider.val
    h01 = h01_slider.val
    
    all_h0, W, all_h1, all_R1, all_C, all_R0 = \
        h0_W_h1_R1_C_R0(w11=w11, w12=w12, A1=np.array([h01]), mask_R=True)

    ind = 0
    h0, h1, R1, C, R0 = all_h0[ind], all_h1[ind], all_R1[ind], all_C[ind], all_R0[ind]
    
    visualize_mc_matrix_transform(W,   IN=h0, ax=ax1, patches=patches_W, annotate_with=("i", "j", "W", "$h_1$"))
    visualize_mc_matrix_transform(C.T, IN=h1, ax=ax2, patches=patches_C, annotate_with=("j", "i", "C", "$h_0$"), c="r")
    
    # print(f"h1: {h1}, \nh0: {h0}\n\n")
    # print(f"W:\n {W}, \n\nC.T:\n {C.T}\n\n distance of basis vectors of C.T:\n {np.sqrt(np.sum((C.T[:, 0] - C.T[:, 1])**2))}\n")
    
    # plot LRP result
    if 'lrp_result' in patches_C: patches_C['lrp_result'].remove()
    patches_C['lrp_result'] = ax.add_patch(Circle(R0, radius=.03, color='orange'))
    
    plot_explicit_biases = False
    ### explicit biases ###
    if plot_explicit_biases:
        W_eb = np.block([[(W - W.min(axis=1, keepdims=True)), np.eye(2)], 
                         [np.zeros((2,4))]])
        W_eb /= W_eb.sum(axis=0, keepdims=True)
        # print("W_eb \n", W_eb, '\n')

        biases = W.min(axis=1)
        biases_mass = biases.sum()
        assert(biases_mass <= 1)

        h0_eb = np.hstack((h0 * (1 - biases_mass), biases))
        print("h0_eb", h0_eb, '\n')

        h1_eb = W_eb @ h0_eb
        print("h1_eb", h1_eb, "\nh1", h1, '\n')

        C_eb = W_eb * h0_eb[None, :]
        C_eb /= C_eb.sum(axis=1, keepdims=True)
        C_eb[np.isnan(C_eb)] = 0
        print("C_eb.T \n", C_eb.T, "\n C.T \n", C.T, '\n')

        print((C.T - C_eb.T[:2, :2]) / h0[:, None])
        print("\n\n")

        R1_eb = h1_eb * (h1_eb == h1_eb.max())
        # print("R1_eb", R1_eb, '\n')

        R0_eb = C_eb.T @ R1_eb
        print("R0_eb", R0_eb, '\n')

        visualize_mc_matrix_transform(C_eb.T[:2, :2], IN=h1_eb[:2], ax=ax2, patches=patches_C_eb, annotate_with=None, c="green")
    
        # plot LRP result
        if 'lrp_result' in patches_C_eb: patches_C_eb['lrp_result'].remove()
        patches_C_eb['lrp_result'] = ax.add_patch(Circle(R0_eb[:2], radius=.03, color='orange'))
    
    fig.canvas.draw_idle()


patches_W = {}
patches_C = {}
patches_C_eb = {}

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(13, 6))

ax1.set_title("Transformation of weigth matrix $W$")
ax2.set_title(
"'Backwards' Transformation of LRP. (LRP-Coefficients matrix $C$) \n\
Its inputs $p(j)$ are determined by the left transformation.")

for ax in (ax1,ax2):
    ax.set_xlim(xmin=-.1, xmax=1.1)
    ax.set_ylim(ymin=-.1, ymax=1.1)

    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)

    # ax.legend()
    

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

valmin, valmax = 0, 1

# control p(2 -> 1):
ax_g = plt.axes([0.04, 0.25, 0.0225, 0.63], facecolor=axcolor)
w11_slider = Slider(
    ax=ax_g,
    label="$p(j_1|i_1)$\n$=W_{1,1}$",
    valmin=valmin,
    valmax=valmax,
    valinit=.6,
    orientation="vertical"
)

# control p(1 -> 1):
ax_h = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
w12_slider = Slider(
    ax=ax_h,
    label="$p(j_1|i_2)$\n$=W_{1,2}$",
    valmin=valmin,
    valmax=valmax,
    valinit=.2,
    orientation="vertical"
)

# control h1_1:
ax_h = plt.axes([0.16, 0.25, 0.0225, 0.63], facecolor=axcolor)
h01_slider = Slider(
    ax=ax_h,
    label="$p(i_1)$",
    valmin=valmin,
    valmax=valmax,
    valinit=.333333,
    orientation="vertical"
)

w11_slider.on_changed(update)
w12_slider.on_changed(update)
h01_slider.on_changed(update)

# Create a `matplotlib.widgets.Button`12to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    w11_slider.reset()
    w12_slider.reset()
    h01_slider.reset()
button.on_clicked(reset)

update(None)
plt.show()