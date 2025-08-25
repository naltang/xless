import numpy as np
import scipy.ndimage as ndimage

class XIntensity:
	ARRAY_DTYPE = np.dtype("<u2")
	ARRAY_HEIGHT = 2560
	ARRAY_WIDTH = 2048

	CORNER_SIZE = 100

	# trimming is required to remove the black borders
	TRIM_TOP = 12
	TRIM_LEFT = 12
	TRIM_RIGHT = 450
	TRIM_BOTTOM = 12

	# these intensity values came from the calibration process
	INTENSITY_RATIO_HIGH_OVER_LOW = 1.6858
	INTENSITY_LOW = 15260
	INTENSITY_HIGH = 25726

	FILENAME_CORRECTION_LOW = "correction_low.csv"
	FILENAME_CORRECTION_HIGH = "correction_high.csv"

	dic_pair = {}

	array_correction_low = None
	array_correction_high = None

	def __init__(self, filename_low=FILENAME_CORRECTION_LOW, filename_high=FILENAME_CORRECTION_HIGH):
		self.read_correction_files(filename_low, filename_high)

	def read_correction_files(self, filename_low, filename_high):
		self.array_correction_low = np.loadtxt(filename_low, delimiter=",", dtype=np.float32)
		self.array_correction_high = np.loadtxt(filename_high, delimiter=",", dtype=np.float32)

	@staticmethod
	def read_raw_file(filename):
		array_1d = np.fromfile(filename, dtype=XIntensity.ARRAY_DTYPE) # little‑endian unsigned short
		if array_1d.size != XIntensity.ARRAY_HEIGHT * XIntensity.ARRAY_WIDTH:
			raise ValueError(f"File {filename} has {array_1d.size} elements, expected {XIntensity.ARRAY_HEIGHT * XIntensity.ARRAY_WIDTH}.")

		array_2d = array_1d.reshape(XIntensity.ARRAY_HEIGHT, XIntensity.ARRAY_WIDTH)	# height, width
		return array_2d

	def read_raw_file_pair(self, filename_low, filename_high):
		array_low = XIntensity.read_raw_file(filename_low)
		array_high = XIntensity.read_raw_file(filename_high)
		self.dic_pair["low"] = array_low
		self.dic_pair["high"] = array_high

	@staticmethod
	def denoise(array):
		array_denoise = ndimage.median_filter(array, size=3) # median of 3x3
		return array_denoise

	@staticmethod
	def crop(array):
		array_crop = array[XIntensity.TRIM_TOP:XIntensity.ARRAY_HEIGHT-XIntensity.TRIM_BOTTOM, XIntensity.TRIM_LEFT:XIntensity.ARRAY_WIDTH-XIntensity.TRIM_RIGHT]
		return array_crop

	@staticmethod
	def uncrop(array, border_value=0):
		array_uncrop = np.zeros((XIntensity.ARRAY_HEIGHT, XIntensity.ARRAY_WIDTH), dtype=array.dtype)
		array_uncrop[:XIntensity.TRIM_TOP, :] = border_value
		array_uncrop[XIntensity.ARRAY_HEIGHT-XIntensity.TRIM_BOTTOM:, :] = border_value
		array_uncrop[:, :XIntensity.TRIM_LEFT] = border_value
		array_uncrop[:, XIntensity.ARRAY_WIDTH-XIntensity.TRIM_RIGHT:] = border_value
		array_uncrop[XIntensity.TRIM_TOP:XIntensity.ARRAY_HEIGHT-XIntensity.TRIM_BOTTOM, XIntensity.TRIM_LEFT:XIntensity.ARRAY_WIDTH-XIntensity.TRIM_RIGHT] = array
		return array_uncrop

	@staticmethod
	def mean_of_4_corners(array):
		# Get the mean intensity of the 4 corners
		array_corner = np.concatenate([
			array[0:XIntensity.CORNER_SIZE, 0:XIntensity.CORNER_SIZE],
			array[0:XIntensity.CORNER_SIZE, -XIntensity.CORNER_SIZE-1:-1],
			array[-XIntensity.CORNER_SIZE-1:-1, 0:XIntensity.CORNER_SIZE],
			array[-XIntensity.CORNER_SIZE-1:-1, -XIntensity.CORNER_SIZE-1:-1]
		])
		return array_corner.mean()

	@staticmethod
	def mean_of_top_half(array):

		array_1d = array.ravel()

		n = array_1d.size
		if n == 0:
			raise ValueError("array must contain at least one element")

		k = (n + 1) // 2

		# Find the threshold value: the k‑th largest element.
		# `np.partition` places the k smallest values in the first k slots,
		# but we want the *k*-th largest, so use n-k as the partition index.
		threshold = np.partition(array_1d, -k)[-k]

		# Keep only elements >= threshold
		top_half = array_1d[array_1d >= threshold]
		return top_half.mean()

	def intensity_correction(self):
		self.dic_pair["low_crop"] = XIntensity.crop(self.dic_pair["low"])
		self.dic_pair["low_denoise"] = XIntensity.denoise(self.dic_pair["low_crop"])
		self.dic_pair["low_flatten"] = self.dic_pair["low_denoise"] * self.array_correction_low
		self.dic_pair["low_corrected"] = self.dic_pair["low_flatten"] / XIntensity.mean_of_top_half(self.dic_pair["low_flatten"]) * XIntensity.INTENSITY_LOW
		self.dic_pair["high_crop"] = XIntensity.crop(self.dic_pair["high"])
		self.dic_pair["high_denoise"] = XIntensity.denoise(self.dic_pair["high_crop"])
		self.dic_pair["high_flatten"] = self.dic_pair["high_denoise"] * self.array_correction_high
		self.dic_pair["high_corrected"] = self.dic_pair["high_flatten"] / XIntensity.mean_of_top_half(self.dic_pair["high_flatten"]) * XIntensity.INTENSITY_HIGH

if __name__ == "__main__":
	x_intensity = XIntensity()
	x_intensity.read_raw_file_pair("data/sample_low.raw", "data/sample_high.raw")
	x_intensity.intensity_correction()
	mean_low = XIntensity.mean_of_top_half(x_intensity.dic_pair["low_corrected"])
	mean_high = XIntensity.mean_of_top_half(x_intensity.dic_pair["high_corrected"])
	print(f"mean low = {mean_low}, mean high = {mean_high}, ratio H/L = {mean_high / mean_low if mean_low != 0 else 0}")