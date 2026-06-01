import numpy as np
import matplotlib.pyplot as plt
from sensor_geometry import SensorGeometry


class Medium:
    def __init__(self, shape, grid_resolution_mm=1.0):
        self.shape = shape
        self.volume = np.zeros(shape)
        self.grid_resolution_mm = grid_resolution_mm
        self.optical_properties = np.array([[0, 0, 1, 1]])

    def add_ball(self, center_mm, radius_mm, val):
        """
        Adds a ball of specified value to a 3D numpy array, given a physical resolution of the grid.

        Parameters:
        -----------
        center_mm : tuple of floats
            The (x, y, z) coordinates of the ball's center in millimeters.
        radius_mm : float
            The radius of the ball in millimeters.
        val : float
            The value with which the ball's volume will be filled.
        grid_resolution_mm : float, optional
            The physical size (in millimeters) of each grid point in the array. Default is 1 mm.
        """

        # flip y axis of center
        center_mm = (center_mm[0], self.shape[1] - center_mm[1], center_mm[2])

        # Convert center and radius from millimeters to grid points
        center = tuple(coord / self.grid_resolution_mm for coord in center_mm)
        radius = radius_mm / self.grid_resolution_mm

        # Create a meshgrid for the array dimensions
        x, y, z = np.ogrid[0 : self.shape[0], 0 : self.shape[1], 0 : self.shape[2]]

        # Calculate the Euclidean distance from each point in the meshgrid to the center
        distances = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )

        # Set the value for points within the specified radius
        self.volume[distances <= radius] = val

    def plot(self, z_slice=None):
        if z_slice is None:
            z_slice = self.shape[0] // 2
        plt.imshow(self.volume[z_slice])
        plt.gca().invert_yaxis()
        plt.show()

    def get_mua(self):
        mua = np.zeros_like(self.volume, dtype=np.float64)
        for i in range(len(self.optical_properties)):
            mua[self.volume == i] = self.optical_properties[i, 0]
        return mua

    def plot_mua(
        self,
        sensors: SensorGeometry = None,
        z_slice=None,
    ):
        if z_slice is None:
            z_slice = self.shape[0] // 2

        mua = self.get_mua()
        mua[self.volume == 0] = np.nan

        plt.imshow(mua[z_slice])
        plt.colorbar(label="mu_a [1/mm]")
        plt.clim(0.018, 0.022)
        if sensors:
            plt.scatter(
                sensors.det_pos[:, -1],
                sensors.det_pos[:, -2],
                c="g",
                marker="o",
                label="detectors",
            )
            plt.scatter(
                sensors.src_pos[:, -1],
                sensors.src_pos[:, -2],
                c="r",
                marker="x",
                label="sources",
            )
            plt.legend()
        plt.gca().invert_yaxis()
        plt.xlabel("mm")
        plt.ylabel("mm")
        plt.show()
