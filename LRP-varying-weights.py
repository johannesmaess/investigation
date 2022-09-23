import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from helpers import x_W_H_C_R0

# The function to be called anytime a slider's value changes
def update(_):
    # perspective by sliders:
    w11 = w11_slider.val
    w12 = w12_slider.val
    
    all_X, W, all_h, all_C, all_R0 = x_W_H_C_R0(w11=w11, w12=w12, A1=A1)

    # plot NN inputs
    h0.set_xdata(A1)
    h0.set_ydata(A1)
    
    h1.set_xdata(A1)
    h1.set_ydata(all_h[:, 0])
    
    
    # plot input layer Relevancies
    r0.set_xdata(A1)
    r0.set_ydata(all_R0[:, 0])
    
    fig.canvas.draw_idle()
    print(W)


fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111)

# scatter = plt.scatter(x_prime,y_prime)
h0, = ax.plot([], [], label='h0', ms=6, color='k', marker='o', ls='')
h1, = ax.plot([], [], label='h1', ms=6, color='r', marker='o', ls='')
r0, = ax.plot([], [], label='r0', ms=6, color='orange', marker='o', ls='')

# plt.vlines(x=[0, 10], ymin=0, ymax=10, color='r')
# plt.hlines(y=[0, 10], xmin=0, xmax=10, color='r')

ax.set_xlim(xmin=-.1, xmax=1.1)
ax.set_ylim(ymin=-.1, ymax=1.1)

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

ax.legend()

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

valmin, valmax = 0, 1

# control p(2 -> 1):
ax_g = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
w12_slider = Slider(
    ax=ax_g,
    label="w12",
    valmin=valmin,
    valmax=valmax,
    valinit=.6,
    orientation="vertical"
)

# control p(1 -> 1):
ax_h = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
w11_slider = Slider(
    ax=ax_h,
    label='w11',
    valmin=valmin,
    valmax=valmax,
    valinit=.2
)

w11_slider.on_changed(update)
w12_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    w11_slider.reset()
    w12_slider.reset()
button.on_clicked(reset)

A1 = np.arange(11) / 10

update(None)
plt.show()