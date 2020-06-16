import matplotlib
import numpy as np

unit_square_mesh = {'delta': 0.1,
                    'x_min': 0,
                    'x_max': 1,
                    'y_min': 0,
                    'y_max': 1}

all_quadrants_mesh = {'delta': 0.1,
                      'x_min': -0.5,
                      'x_max': 0.5,
                      'y_min': -0.5,
                      'y_max': 0.5}


def setup_plot(T, mesh_properties=unit_square_mesh, square_axes=False,
               plot_columns=False, plot_eigenvectors=False):
    """
    Setup the plot and axes for animating a linear transformation T.

    If asked, plot the columns (aka the images of the basis vectors)
    and the eigenvectors (but only if the eigvals are real and non-zero).

    Parameters
    ----------
    T        : 2x2 matrix representing a linear transformation
    mesh_properties : dictionary that defines properties of meshgrid of points
                        that will be plotted and transformed.
                        needs to have the following five properties:
                        'delta' - mesh spacing
                        '{x,y}{Min,Max}' - minium/maximum value on x/y axis
    square_axes : if False, size the axes so that they contain starting
                    and ending location of each point in grid.
                 if True, size the axes so that they are square and contain
                    starting and ending location of each point in mesh.
    plot_columns : if True, plot the columns of the transformation so that we can see
                        where the basis vectors end up
    plot_eigenvectors: if true, plot the eigenvectors of the transformation

    Returns
    -------
    returns are meant to be consumed by animate_transformation

    scatter   : a PathCollection with all of the points in the meshgrid
    f         : matplotlib figure containing axes
    ax        : matplotlib axes containing scatter
    animate   : callable for use with matplotlib.FuncAnimation
    """
    T = np.asarray(T)

    xs, ys = make_mesh(mesh_properties)
    colors = np.linspace(0, 1, num=xs.shape[0] * xs.shape[1])

    with matplotlib.style.context("dark_background"):
        fig = matplotlib.figure.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

    scatter = plot_mesh(ax, xs, ys, colors)

    plot_vectors = [plot_columns, plot_eigenvectors]

    not_zeros = not(np.all(T == np.zeros(2)))

    if (any(plot_vectors) & not_zeros):
        plot_interesting_vectors(T, ax, *plot_vectors)

    start, end, delta = compute_trajectories(T, scatter)

    mn, mx = calculate_axis_bounds(start, end)

    if square_axes:
        lim = max(abs(mn), abs(mx))
        mn, mx = -lim, lim

    draw_coordinate_axes(mn, mx, ax=ax)
    set_axes_lims(mn, mx, ax=ax)

    offsets = precompute_animation(T, scatter)
    n_frames = len(offsets)

    def animate(ii):
        scatter.set_offsets(offsets[ii])

    return fig, ax, animate, n_frames


def make_mesh(mesh_props):
    num_dimensions = 2

    mins = (mesh_props['x_min'], mesh_props['y_min'])
    maxs = (mesh_props['x_max'], mesh_props['y_max'])
    delta = mesh_props['delta']

    for idx in range(num_dimensions):
        assert mins[idx] < maxs[idx], "min can't be bigger than max!"

    ranges = [np.arange(mins[idx], maxs[idx] + delta, delta)
              for idx in range(num_dimensions)]

    xs, ys = np.meshgrid(*ranges)

    return xs, ys


def plot_mesh(ax, xs, ys, colors):
    h = ax.scatter(xs.flatten(), ys.flatten(),
                   alpha=0.7, edgecolor='none',
                   s=36, linewidth=2,
                   zorder=6,
                   c=colors, cmap='hot')

    return h


def plot_vector(v, ax, color, label=None):
    return ax.arrow(0, 0, v[0], v[1], zorder=5,
                    linewidth=6, color=color, head_width=0.1, label=label)


def plot_interesting_vectors(T, ax, columns=False, eigs=True):
    arrows = []
    labels = []

    if columns:
        arrows += [plot_vector(column, ax, 'hotpink')
                   for column in T.T]
        arrows = [arrows[0]]
        labels += ["a basis vector lands here"]
    if eigs:
        eigenvalues, eigenvectors = np.linalg.eig(T)
        eigen_list = [(eigenvalue, eigenvector) for eigenvalue, eigenvector
                      in zip(eigenvalues, eigenvectors.T)
                      if eigenvalue != 0
                      and not(np.iscomplex(eigenvalue))
                      ]
        if eigen_list:
            eigen_arrows = [
                    plot_vector(np.real(element[1]), ax, '#53fca1', label='special vectors')
                    for element in eigen_list]
            eigen_arrows = [eigen_arrows[0]]
            labels += ["this is a special (aka eigen) vector"]
            arrows += eigen_arrows
        else:
            print("eigenvalues are all nonreal or 0")
    ax.legend(arrows, labels, loc=[0, 0.5],
              bbox_to_anchor=(0, 1.01),
              ncol=1, prop={'weight': 'bold'})
    return


def compute_trajectories(T, scatter):

    starting_positions = scatter.get_offsets()
    ending_positions = np.dot(T, starting_positions.T).T
    delta_positions = ending_positions-starting_positions

    return starting_positions, ending_positions, delta_positions


def set_axes_lims(mn, mx, ax):

    ax.set_ylim([mn, mx])
    ax.set_xlim([mn, mx])

    return


def calculate_axis_bounds(starting_positions, ending_positions, buffer_factor=1.1):
    # axis bounds to include starting and ending positions of each point

    mn = buffer_factor * min(np.min(starting_positions), np.min(ending_positions))
    mx = buffer_factor * max(np.max(starting_positions), np.max(ending_positions))

    if mn == 0:
        mn -= 0.1
    if mx == 0:
        mx += 0.1

    return mn, mx


def draw_coordinate_axes(mn, mx, ax):

    ax.hlines(0, mn, mx, zorder=4, linewidth=4, color='grey')
    ax.vlines(0, mn, mx, zorder=4, linewidth=4, color='grey')

    return


def make_rotation(theta):
    rotation_matrix = [[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]]
    return np.asarray(rotation_matrix)


def precompute_animation(T, scatter, delta_t=0.01):

    T = np.asarray(T)

    start, _, delta = compute_trajectories(T, scatter)

    Id = np.eye(2)

    ts = np.arange(0, 1 + delta_t, delta_t)

    not_zeros = not(np.all(T == np.zeros(2)))

    offsets = [scatter.get_offsets()]

    if ((T[0, 0] == T[1, 1]) & (T[0, 1] == -1 * T[1, 0])) & \
            not_zeros:

        z = complex(T[0, 0], T[1, 0])
        dz = z ** (1 / len(ts))

        dT = [[dz.real, -dz.imag], [dz.imag, dz.real]]
        for idx, t in enumerate(ts):
            dT_toN = np.linalg.matrix_power(dT, idx+1)
            offsets.append(np.dot(dT_toN, start.T).T)

    else:
        for idx, t in enumerate(ts):
            offsets.append(t * (np.dot(T, start.T).T) +
                           (1 - t) * np.dot(Id, start.T).T)

    return offsets
