import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asp_plot.utils import ColorBar, Raster, Plotter


class StereoPlotter(Plotter):
    def __init__(
        self, directory, stereo_directory, reference_dem, out_dem_gsd=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory
        self.reference_dem = reference_dem
        self.out_dem_gsd = out_dem_gsd

        try:
            self.left_ortho_sub_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*-L_sub.tif")
            )[0]
            self.right_ortho_sub_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*-R_sub.tif")
            )[0]
            self.left_ortho_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*-L.tif")
            )[0]
        except:
            raise ValueError(
                "Could not find L, L-sub, and R-sub images in stereo directory"
            )

        try:
            self.match_point_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*.match")
            )[0]
        except:
            raise ValueError("Could not find *.match file in stereo directory")

        try:
            self.disparity_sub_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*-D_sub.tif")
            )[0]
        except:
            raise ValueError("Could not find *.match file in stereo directory")

        try:
            try:
                self.dem_fn = glob.glob(
                    os.path.join(
                        self.directory,
                        self.stereo_directory,
                        f"*-DEM_{self.out_dem_gsd}m.tif",
                    )
                )[0]
            except:
                self.dem_fn = glob.glob(
                    os.path.join(
                        self.directory,
                        self.stereo_directory,
                        f"*{self.out_dem_gsd}m-DEM.tif",
                    )
                )[0]
        except:
            raise ValueError("Could not find DEM file in stereo directory")

        try:
            self.intersection_error_rn = glob.glob(
                os.path.join(
                    self.directory, self.stereo_directory, "*-IntersectionErr.tif"
                )
            )[0]
        except:
            raise ValueError(
                "Could not find IntersectionError file in stereo directory"
            )

    def get_hillshade(self):
        hs_fn = os.path.splitext(self.dem_fn)[0] + "_hs.tif"
        if os.path.exists(hs_fn):
            hs = Raster(hs_fn).read_array()
        else:
            hs = Raster(self.dem_fn).hillshade()
        return hs

    def get_diff_vs_reference(self):
        diff_fn = glob.glob(
            os.path.join(
                self.directory,
                self.stereo_directory,
                f"*DEM*diff.tif",
            )
        )[0]
        if os.path.exists(diff_fn):
            diff = Raster(diff_fn).read_array()
        else:
            diff = Raster(self.dem_fn).compute_difference(self.reference_dem)
        return diff

    def read_ip_record(self):
        x, y = np.frombuffer(mf.read(8), dtype=np.float32)
        xi, yi = np.frombuffer(mf.read(8), dtype=np.int32)
        orientation, scale, interest = np.frombuffer(mf.read(12), dtype=np.float32)
        polarity, = np.frombuffer(mf.read(1), dtype=bool)
        octave, scale_lvl = np.frombuffer(mf.read(8), dtype=np.uint32)
        ndesc, = np.frombuffer(mf.read(8), dtype=np.uint64)
        desc = np.frombuffer(mf.read(int(ndesc * 4)), dtype=np.float32)
        iprec = [x, y, xi, yi, orientation, 
                scale, interest, polarity, 
                octave, scale_lvl, ndesc]
        iprec.extend(desc)
        return iprec

    def get_match_point_df(self):
        out_csv = os.path.splitext(self.match_point_fn)[0] + ".csv"
        if not os.path.exists(out_csv):
            with open(self.match_point_fn, "rb") as mf, open(out_csv, "w") as out:
                size1 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
                size2 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
                out.write("x1 y1 x2 y2\n")
                im1_ip = [read_ip_record(mf) for i in range(size1)]
                im2_ip = [read_ip_record(mf) for i in range(size2)]
                for i in range(len(im1_ip)):
                    out.write("{} {} {} {}\n".format(im1_ip[i][0], 
                                                    im1_ip[i][1], 
                                                    im2_ip[i][0], 
                                                    im2_ip[i][1]))
        
        match_point_df = pd.read_csv(out_csv, delimiter=r"\s+")
        return match_point_df

    def plot_match_points(self):
        match_point_df = self.get_match_point_df()

        full_gsd = Raster(self.left_ortho_fn).get_gsd()
        sub_gsd = Raster(self.left_ortho_sub_fn).get_gsd()
        rescale_factor = sub_gsd / full_gsd

        fig, axa = plt.subplots(1, 2, figsize=(10, 5))

        left_image = Raster(self.left_ortho_sub_fn).read_array()
        right_image = Raster(self.right_ortho_sub_fn).read_array()

        self.plot_array(ax=axa[0], array=left_image)
        self.plot_array(ax=axa[1], array=right_image)

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

        fig.suptitle(self.title, size=10)
        plt.tight_layout()
