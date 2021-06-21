import numpy as np
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks_cwt

def find_stalks(corn_pixel_coord, im_h, im_w):
	'''
	find corn stalks position from segmented corn pixel coordinates
	Input
		corn_pixel_coord -- (m, 2)
		im_h -- image.shape[0]
		im_w -- image.shape[1]
	'''
	# filter top 100 pixels
	corn_pixel_coord = corn_pixel_coord[corn_pixel_coord[:,1] > 100]

	m = corn_pixel_coord.shape[0]

	count = csr_matrix((np.ones(m), (corn_pixel_coord[:,1], corn_pixel_coord[:,0])), shape=(im_h, im_w))
	hist = np.array(count.sum(axis=0)).reshape(-1)

	assert hist.shape[0] == im_w

	peakind = find_peaks_cwt(hist, np.arange(35,60))
	return peakind