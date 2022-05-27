from matplotlib import pyplot as plt, patches


def show_observation(observation, r: float, name: str = "solution"):
    fig, ax = plt.subplots()
    plt.xlim(-r * 1.25, r * 1.25)
    plt.ylim(-r * 1.25, r * 1.25)

    plt.grid(linestyle='--')
    ax.set_aspect(1)
    ax.add_artist(plt.Circle((0, 0), r, color='r', fill=False))

    for rect in observation.rectangles:
        rect_img = patches.Rectangle((rect.x, rect.y - rect.height), rect.width, rect.height, edgecolor="b")
        ax.add_patch(rect_img)

    plt.savefig(f"cutting-stock/visualisation/{name}")
