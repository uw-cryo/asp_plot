import matplotlib.pyplot as plt
from asp_plot.utils import Raster, Plotter


class ScenePlotter(Plotter):
    def __init__(self, left_ortho_fn, right_ortho_fn, **kwargs):
        super().__init__(**kwargs)
        self.left_ortho_fn = left_ortho_fn
        self.right_ortho_fn = right_ortho_fn

    def plot_orthos(self, title=None):
        f, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        f.suptitle(title, size=10)
        axa = axa.ravel()

        if self.left_ortho_fn:
            ortho_ma = Raster(self.left_ortho_fn).read_array()
            self.plot_array(ax=axa[0], array=ortho_ma)
            axa[0].set_title("Left image")
        else:
            axa[0].axis("off")

        if self.right_ortho_fn:
            ortho_ma = Raster(self.right_ortho_fn).read_array()
            self.plot_array(ax=axa[1], array=ortho_ma)
            axa[1].set_title("Right image")
        else:
            axa[1].axis("off")

        f.tight_layout()
        plt.show()
