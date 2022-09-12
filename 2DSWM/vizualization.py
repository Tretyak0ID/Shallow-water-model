import matplotlib.pyplot as plt
from matplotlib import animation, cm


def animate_imshow(f, domain, anim_speed):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = plt.imshow(f[0], animated=True, interpolation='bicubic', cmap='seismic')

    def animate(n):
        frame = anim_speed * n
        im.set_array(f[frame])
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=len(f) // anim_speed, interval=1, repeat=False, blit=True)
    plt.show()
    return anim


def animate_surf(f, domain, anim_speed=1):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

    ax.plot_surface(domain.xx, domain.yy, f[0])
    height = f[0].max() - f[0].min()
    zmin = f[0].min() - 0.5 * height
    zmax = f[0].max() + 0.5 * height
    ax.set_zlim(zmin, zmax)

    def animate(n):
        frame = anim_speed * n

        ax.cla()
        ax.plot_surface(domain.xx, domain.yy, f[frame])

        ax.set_zlim(zmin, zmax)
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=len(f) // anim_speed, interval=1, repeat=False, blit=True)
    plt.show()

    return anim


def surf_field(f, domain):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(domain.xx, domain.yy, f)
    plt.show()

    return surf


def imshow_field(f, domain):
    fig, ax = plt.subplots()
    im = plt.imshow(f, animated=True, interpolation='bilinear', cmap='seismic')
    plt.show()
    return im

def hv_field_plot(domain, state):
    fig, ax = plt.subplots()
    ax.streamplot(domain.xx, domain.yy, state.u, state.v, color = 'b')