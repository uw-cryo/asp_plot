import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar

from asp_plot.utils import ColorBar, Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class StereoPlotter(Plotter):
    def __init__(
        self, directory, stereo_directory, reference_dem, out_dem_gsd=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory
        self.reference_dem = reference_dem
        self.out_dem_gsd = out_dem_gsd

        self.full_directory = os.path.join(self.directory, self.stereo_directory)
        self.left_ortho_sub_fn = glob_file(self.full_directory, "*-L_sub.tif")
        self.right_ortho_sub_fn = glob_file(self.full_directory, "*-R_sub.tif")
        self.left_ortho_fn = glob_file(self.full_directory, "*-L.tif")
        self.match_point_fn = glob_file(self.full_directory, "*.match")
        self.disparity_sub_fn = glob_file(self.full_directory, "*-D_sub.tif")
        self.disparity_fn = glob_file(self.full_directory, "*-D.tif")
        self.dem_fn = glob_file(
            self.full_directory,
            f"*-DEM_{self.out_dem_gsd}m.tif",
            f"*{self.out_dem_gsd}m-DEM.tif",
        )
        self.intersection_error_fn = glob_file(
            self.full_directory, "*-IntersectionErr.tif"
        )

    def get_diff_vs_reference(self):
        diff_fn = glob_file(self.full_directory, "*DEM*diff.tif")
        if diff_fn:
            diff = Raster(diff_fn).read_array()
        else:
            if self.dem_fn is None or self.reference_dem is None:
                return None
            else:
                diff = Raster(self.dem_fn).compute_difference(self.reference_dem)
        return diff

    def read_ip_record(self, match_file):
        x, y = np.frombuffer(match_file.read(8), dtype=np.float32)
        xi, yi = np.frombuffer(match_file.read(8), dtype=np.int32)
        orientation, scale, interest = np.frombuffer(
            match_file.read(12), dtype=np.float32
        )
        (polarity,) = np.frombuffer(match_file.read(1), dtype=bool)
        octave, scale_lvl = np.frombuffer(match_file.read(8), dtype=np.uint32)
        (ndesc,) = np.frombuffer(match_file.read(8), dtype=np.uint64)
        desc = np.frombuffer(match_file.read(int(ndesc * 4)), dtype=np.float32)
        iprec = [
            x,
            y,
            xi,
            yi,
            orientation,
            scale,
            interest,
            polarity,
            octave,
            scale_lvl,
            ndesc,
        ]
        iprec.extend(desc)
        return iprec

    def get_match_point_df(self):
        out_csv = (
            os.path.splitext(self.match_point_fn)[0] + ".csv"
            if self.match_point_fn
            else None
        )
        if self.match_point_fn and not os.path.exists(out_csv):
            with (
                open(self.match_point_fn, "rb") as match_file,
                open(out_csv, "w") as out,
            ):
                size1 = np.frombuffer(match_file.read(8), dtype=np.uint64)[0]
                size2 = np.frombuffer(match_file.read(8), dtype=np.uint64)[0]
                out.write("x1 y1 x2 y2\n")
                im1_ip = [self.read_ip_record(match_file) for i in range(size1)]
                im2_ip = [self.read_ip_record(match_file) for i in range(size2)]
                for i in range(len(im1_ip)):
                    out.write(
                        "{} {} {} {}\n".format(
                            im1_ip[i][0], im1_ip[i][1], im2_ip[i][0], im2_ip[i][1]
                        )
                    )

        return (
            pd.read_csv(out_csv, delimiter=r"\s+")
            if out_csv and os.path.exists(out_csv)
            else None
        )

    def plot_match_points(self, save_dir=None, fig_fn=None):
        match_point_df = self.get_match_point_df()

        fig, axa = plt.subplots(1, 2, figsize=(10, 5))

        if (
            self.left_ortho_sub_fn
            and self.right_ortho_sub_fn
            and match_point_df is not None
        ):
            full_gsd = Raster(self.left_ortho_fn).get_gsd()
            sub_gsd = Raster(self.left_ortho_sub_fn).get_gsd()
            rescale_factor = sub_gsd / full_gsd

            left_image = Raster(self.left_ortho_sub_fn).read_array()
            right_image = Raster(self.right_ortho_sub_fn).read_array()

            self.plot_array(ax=axa[0], array=left_image, cmap="gray", add_cbar=False)
            axa[0].set_title(f"Left image (n={match_point_df.shape[0]})")
            self.plot_array(ax=axa[1], array=right_image, cmap="gray", add_cbar=False)
            axa[1].set_title("Right image")

            axa[0].scatter(
                match_point_df["x1"] / rescale_factor,
                match_point_df["y1"] / rescale_factor,
                color="r",
                marker="o",
                facecolor="none",
                s=1,
            )
            axa[0].set_aspect("equal")

            axa[1].scatter(
                match_point_df["x2"] / rescale_factor,
                match_point_df["y2"] / rescale_factor,
                color="r",
                marker="o",
                facecolor="none",
                s=1,
            )
            axa[1].set_aspect("equal")
        else:
            axa[0].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[0].transAxes,
            )
            axa[1].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[1].transAxes,
            )

        fig.suptitle(self.title, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_disparity(
        self, unit="pixels", remove_bias=True, quiver=True, save_dir=None, fig_fn=None
    ):
        if unit not in ["pixels", "meters"]:
            raise ValueError(
                "\n\nUnit for disparity plot must be either 'pixels' or 'meters'\n\n"
            )

        fig, axa = plt.subplots(1, 3, figsize=(10, 3), dpi=220)
        fig.suptitle(self.title, size=10)

        if self.disparity_sub_fn and self.disparity_fn:
            raster = Raster(self.disparity_sub_fn)
            sub_gsd = raster.get_gsd()
            dx = raster.read_array(b=1)
            dy = raster.read_array(b=2)

            full_gsd = Raster(self.disparity_fn).get_gsd()
            rescale_factor = sub_gsd / full_gsd
            dx *= rescale_factor
            dy *= rescale_factor

            if unit == "meters":
                dx *= full_gsd
                dy *= full_gsd

            if remove_bias:
                dx_offset = np.ma.median(dx)
                dy_offset = np.ma.median(dy)
                dx -= dx_offset
                dy -= dy_offset

            dm = np.sqrt(abs(dx**2 + dy**2))
            clim = ColorBar(symm=True).get_clim(dm)

            self.plot_array(
                ax=axa[0], array=dx, cmap="RdBu", clim=clim, cbar_label=unit
            )
            self.plot_array(
                ax=axa[1], array=dy, cmap="RdBu", clim=clim, cbar_label=unit
            )
            self.plot_array(
                ax=axa[2], array=dm, cmap="inferno", clim=(0, clim[1]), cbar_label=unit
            )

            if quiver:
                fraction = 30
                stride = max(1, int(dx.shape[1] / fraction))
                iy, ix = np.indices(dx.shape)[:, ::stride, ::stride]
                dx_q = dx[::stride, ::stride]
                dy_q = dy[::stride, ::stride]
                axa[2].quiver(ix, iy, dx_q, dy_q, color="white")

            scalebar = ScaleBar(sub_gsd)
            axa[0].add_artist(scalebar)
            axa[0].set_title("x offset")
            axa[1].set_title("y offset")
            axa[2].set_title("offset magnitude")
        else:
            for ax in axa:
                ax.text(
                    0.5,
                    0.5,
                    "One or more required\nfiles are missing",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_dem_results(
        self, el_clim=None, ie_clim=None, diff_clim=None, save_dir=None, fig_fn=None
    ):
        print("Plotting DEM results. This can take a minute for large inputs.")
        fig, axa = plt.subplots(1, 3, figsize=(10, 3), dpi=220)
        fig.suptitle(self.title, size=10)

        if self.dem_fn:
            raster = Raster(self.dem_fn)
            dem = raster.read_array()
            gsd = raster.get_gsd()

            hs = raster.hillshade()
            self.plot_array(ax=axa[0], array=hs, cmap="gray", add_cbar=False)
            self.plot_array(
                ax=axa[0],
                array=dem,
                clim=el_clim,
                cmap="viridis",
                cbar_label="Elevation (m HAE)",
                alpha=0.5,
            )

            axa[0].set_title("Stereo DEM")
            scalebar = ScaleBar(gsd)
            axa[0].add_artist(scalebar)
        else:
            axa[0].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[0].transAxes,
            )

        if self.intersection_error_fn:
            ie = Raster(self.intersection_error_fn).read_array()
            self.plot_array(
                ax=axa[1],
                array=ie,
                clim=ie_clim,
                cmap="inferno",
                cbar_label="Distance (m)",
            )
            axa[1].set_title("Triangulation intersection error")
        else:
            axa[1].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[1].transAxes,
            )

        diff = self.get_diff_vs_reference()
        if diff is not None:
            if diff_clim is None:
                diff_clim = ColorBar(perc_range=(2, 98), symm=True).get_clim(diff)

            self.plot_array(
                ax=axa[2],
                array=diff,
                clim=diff_clim,
                cmap="RdBu",
                cbar_label="Elevation diff. (m)",
            )
            axa[2].set_title(
                f"Difference with reference DEM:\n{self.reference_dem.split('/')[-1]}"
            )
        else:
            axa[2].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[2].transAxes,
            )

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
