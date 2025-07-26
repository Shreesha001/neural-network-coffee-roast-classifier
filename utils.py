import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid
import matplotlib.colors as colors
from matplotlib import cm

# Optional: define fallback colors in case dlc is not available
dlc = {
    "dldarkblue": "#003f5c",
    "dlpurple": "#7a5195"
}

def plt_layer(X, Y, W1, b1, norm_l):
    Y = Y.reshape(-1,)
    fig, ax = plt.subplots(1, W1.shape[1], figsize=(16, 4))

    for i in range(W1.shape[1]):
        layerf = lambda x: sigmoid(np.dot(norm_l(x), W1[:, i]) + b1[i])
        plt_prob(ax[i], layerf)
        
        ax[i].scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")
        ax[i].scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
                      edgecolors=dlc["dldarkblue"], linewidth=1, label="Bad Roast")

        tr = np.linspace(175, 260, 50)
        ax[i].plot(tr, (-3/85) * tr + 21, color=dlc["dlpurple"], linewidth=2)
        ax[i].axhline(y=12, color=dlc["dlpurple"], linewidth=2)
        ax[i].axvline(x=175, color=dlc["dlpurple"], linewidth=2)
        
        ax[i].set_title(f"Layer 1, unit {i}")
        ax[i].set_xlabel("Temperature \n(Celsius)", size=12)
    
    ax[0].set_ylabel("Duration \n(minutes)", size=12)
    plt.tight_layout()
    plt.show()

def plt_prob(ax, fwb):
    """Plots a decision boundary but includes shading to indicate the probability"""
    x0_space = np.linspace(150, 285, 40)
    x1_space = np.linspace(11.5, 15.5, 40)

    tmp_x0, tmp_x1 = np.meshgrid(x0_space, x1_space)
    z = np.zeros_like(tmp_x0)

    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x1.shape[1]):
            x = np.array([[tmp_x0[i, j], tmp_x1[i, j]]])
            z[i, j] = fwb(x)

    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)

    pcm = ax.pcolormesh(tmp_x0, tmp_x1, z,
                        norm=cm.colors.Normalize(vmin=0, vmax=1),
                        cmap=new_cmap, shading='nearest', alpha=0.9)
    ax.figure.colorbar(pcm, ax=ax)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncates a color map"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_output_neuron_3D(W, b):
    """
    Visualizes how the final output neuron responds to 3 inputs (from hidden units).
    Input:
        W : numpy array of shape (3,1) — weights from 3 units to final neuron
        b : scalar bias — bias for the final neuron
    Output:
        3D scatter plot where color indicates output probability (after sigmoid)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Grid of values for 3 inputs (u0, u1, u2), each in [0, 1]
    steps = 10
    x_vals = np.linspace(0., 1., steps)
    y_vals = np.linspace(0., 1., steps)
    z_vals = np.linspace(0., 1., steps)

    # Create 3D mesh
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    d = np.zeros((steps, steps, steps))

    # Compute sigmoid output for each (x, y, z)
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                v = np.array([x[i,j,k], y[i,j,k], z[i,j,k]])
                linear_output = np.dot(v, W[:,0]) + b
                d[i,j,k] = 1 / (1 + np.exp(-linear_output))  # Sigmoid manually

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap('Blues')
    pcm = ax.scatter(x, y, z, c=d, cmap=cmap, alpha=1)

    # Labels and view
    ax.set_xlabel("Unit 0 Output")
    ax.set_ylabel("Unit 1 Output")
    ax.set_zlabel("Unit 2 Output")
    ax.set_title("Final Output Neuron Response")
    ax.view_init(elev=30, azim=-120)
    fig.colorbar(pcm, ax=ax, label='Output Probability')

    plt.show()

dlc = {
    "dldarkblue": "#2C5282",
    "dlpurple": "#6B46C1"
}

def plt_roast(X, Y):
    Y = Y.reshape(-1,)  # ensure Y is a 1D array
    fig, ax = plt.subplots(figsize=(6,4))

    # Scatter plot for Good Roast (Y==1)
    ax.scatter(
        X[Y == 1, 0], X[Y == 1, 1],
        s=70, marker='x', c='red',
        label="Good Roast"
    )

    # Scatter plot for Bad Roast (Y==0)
    ax.scatter(
        X[Y == 0, 0], X[Y == 0, 1],
        s=100, marker='o', facecolors='none',
        edgecolors=dlc["dldarkblue"], linewidth=1,
        label="Bad Roast"
    )

    # Custom decision boundary or guide lines
    tr = np.linspace(175, 260, 50)
    ax.plot(tr, (-3 / 85) * tr + 21, color=dlc["dlpurple"], linewidth=1)
    ax.axhline(y=12, color=dlc["dlpurple"], linewidth=1)
    ax.axvline(x=175, color=dlc["dlpurple"], linewidth=1)

    # Labels and Title
    ax.set_title("Coffee Roasting", fontsize=16)
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Duration (minutes)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True)

    plt.show()

def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))
