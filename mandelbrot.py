import numpy as np
from numba import jit
#DISPLAY WITH matplotlib and jupiter
from matplotlib import pyplot as plt
from matplotlib import colors


@jit
def mandelbrot(z, maxi, horizon, log_horizon):
    c = z
    for n in range(maxi):
        az = abs(z)
        if az > horizon:
            return n - np.log(np.log(az))/np.log(2) + log_horizon
        z = z*z + c
    return 0

@jit
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxi):
    horizon = 2.0 ** 40
    log_horizon = np.log(np.log(horizon))/np.log(2)
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j], maxi, horizon, log_horizon);
    return (r1, r2, n3)


#Function to save image
image_counter = 32

def save_img(fig):
    global image_counter
    filename = "mandelbrot_%d.png" % image_counter
    fig.savefig(filename)

def mandelbrot_image(xmin, xmax, ymin, ymax, width=10, height=10, maxi=8192, cmap='jet', gamma=0.3):
    dpi = 72
    i_width = dpi * width
    i_height = dpi * height;
    x,y,z = mandelbrot_set(xmin, xmax, ymin, ymax, i_width, i_height, maxi)

    fig, ax = plt.subplots(figsize=(width, height), dpi=72)
    ticks = np.arange(0, i_width, 3*dpi)
    x_ticks = xmin + (xmax - xmin) * ticks / i_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax - ymin) * ticks / i_height
    plt.yticks(ticks, y_ticks)

    norm = colors.PowerNorm(gamma)
    ax.imshow(z.T, cmap=cmap, origin='lower', norm=norm)

    save_img(fig)


#Execute
#mandelbrot_image(-2.0, 0.5, -1.25, 1.25)
#mandelbrot_image(-0.8, 0.7, 0, 0.1, cmap="hot")
#mandelbrot_image(-0.74877, -0.748725, 0.06505, 0.06509, cmap="hot")
