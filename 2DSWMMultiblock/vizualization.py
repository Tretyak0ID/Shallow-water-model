import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.interpolate
from matplotlib import animation, cm
import scipy as sp

def animate_trisurf(f, domains, anim_speed=1, title = None, save = False, Name = None):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    tri = mtri.Triangulation(x, y)

    z = []
    for snapshot in f:
        z.append(np.concatenate([snapshot[k].flatten() for k in range(len(domains))]))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

    ax.plot_trisurf(x, y, z[0], triangles=tri.triangles)
    height = z[0].max() - z[0].min()
    zmin = z[0].min() - 0.5 * height
    zmax = z[0].max() + 0.5 * height
    ax.set_zlim(zmin, zmax)

    def animate(n):
        frame = anim_speed * n

        ax.cla()
        ax.plot_trisurf(x, y, z[frame], triangles=tri.triangles)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        plt.title(Name)

        ax.set_zlim(zmin, zmax)
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=len(f) // anim_speed, interval=1, repeat=False)
    plt.show()

    if (save == True):
        anim.save(title, writer=None, fps=30)

    return anim


def tripcolor_field(f, domains):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    tri = mtri.Triangulation(x, y)

    z = np.concatenate([f[k].flatten() for k in range(len(domains))])

    fig, ax = plt.subplots(figsize=(8, 6))

    pic = ax.tripcolor(tri, z, shading='flat', cmap='seismic')

    return pic


def animate_tripcolor(f, domains, anim_speed=1):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    tri = mtri.Triangulation(x, y)

    z = []
    for snapshot in f:
        z.append(np.concatenate([snapshot[k].flatten() for k in range(len(domains))]))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.tripcolor(tri, z[0], shading='flat', cmap='seismic')

    def animate(n):
        frame = anim_speed * n

        ax.cla()
        ax.tripcolor(tri, z[frame], shading='flat', cmap='seismic')
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=len(f) // anim_speed, interval=10, repeat=False, blit=True)
    plt.show()

    return anim


def animate_interpolated_surf(f, domains, anim_speed=1):
    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])

    xs = min([domain.xs for domain in domains])
    xe = max([domain.xe for domain in domains])

    ys = min([domain.ys for domain in domains])
    ye = max([domain.ye for domain in domains])

    dx = min([domain.dx for domain in domains])
    dy = min([domain.dy for domain in domains])

    grid_x, grid_y = np.mgrid[xs:xe:dx, ys:ye:dy]

    grid_z = []
    for snapshot in f:
        values = np.concatenate([snapshot[k].flatten() for k in range(len(domains))])
        grid_z.append(scipy.interpolate.griddata((x, y), values, (grid_x, grid_y), method='linear'))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(grid_x, grid_y, grid_z[0])
    height = grid_z[0].max() - grid_z[0].min()
    zmin = grid_z[0].min() - 0.5 * height
    zmax = grid_z[0].max() + 0.5 * height
    ax.set_zlim(zmin, zmax)

    def animate(n):
        frame = anim_speed * n

        ax.cla()
        ax.plot_surface(grid_x, grid_y, grid_z[frame])

        ax.set_zlim(zmin, zmax)
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=len(f), interval=1, repeat=False, blit=True)
    plt.show()

    return anim


def surf_interpolated_field(f, domains, surf_nx=100, surf_ny=100):
    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    z = np.concatenate([f[k].flatten() for k in range(len(domains))])

    xs = min([domain.xs for domain in domains])
    xe = max([domain.xe for domain in domains])

    ys = min([domain.ys for domain in domains])
    ye = max([domain.ye for domain in domains])

    dx = min([domain.dx for domain in domains])
    dy = min([domain.dy for domain in domains])

    grid_x, grid_y = np.mgrid[xs:xe:dx, ys:ye:dy]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    grid_z = scipy.interpolate.griddata((x, y), z, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap = 'GnBu')
    plt.colorbar(surf)
    plt.show()

    return surf


def imshow_interpolated_field(f, domains, savefig = False, title = None, surf_nx=100, surf_ny=100):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    z = np.concatenate([f[k].flatten() for k in range(len(domains))])

    xs = min([domain.xs for domain in domains])
    xe = max([domain.xe for domain in domains])

    ys = min([domain.ys for domain in domains])
    ye = max([domain.ye for domain in domains])

    dx = min([domain.dx for domain in domains])
    dy = min([domain.dy for domain in domains])

    grid_x, grid_y = np.mgrid[xs:xe:dx, ys:ye:dy]


    grid_z = scipy.interpolate.griddata((x, y), z, (grid_x, grid_y), method='linear')

    imshow = plt.imshow(grid_z.T, cmap='seismic')
    plt.title(title)

    return imshow


def animate_imshow(f, domains, anim_speed=1, title = None, save = False, Name = None):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    tri = mtri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = imshow_interpolated_field(f[0], domains)

    def animate(n):
        frame = anim_speed * n
        ax.cla()
        im.set_array(f[frame], domains)
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=len(f) // anim_speed, interval=1, repeat=False)
    plt.show()

    if (save == True):
        anim.save(title, writer=None, fps=30)

    return anim


def tripcolor_field(f, domains):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    tri = mtri.Triangulation(x, y)

    z = np.concatenate([f[k].flatten() for k in range(len(domains))])

    fig, ax = plt.subplots(figsize=(8, 6))

    pic = ax.tripcolor(tri, z, shading='flat', cmap='seismic')

    return pic


def trisurf_field(f, domains):

    x = np.concatenate([domain.xx.flatten() for domain in domains])
    y = np.concatenate([domain.yy.flatten() for domain in domains])
    z = np.concatenate([f[k].flatten() for k in range(len(domains))])
    tri = mtri.Triangulation(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    trisurf = ax.plot_trisurf(x, y, z, triangles = tri.triangles)#, cmap=plt.cm.Spectral)
    plt.show()

    return trisurf