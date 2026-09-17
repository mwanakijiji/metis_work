import matplotlib.pyplot as plt
from astropy.io import fits

'''
Fundamental reqs.:

1. METIS-1408: Quality and alignment of the optical components within Mid-infrared ELT Imager and
Spectrograph (METIS) shall provide diffraction limited performance (Strehl ≥ 80 %)
at λ > 3μm in all modes over the entire FOV.
2. METIS-1409: The Instrument Wavefront Error (WFE) shall satisfy the diffraction limit requirement
(Strehl>0.8) at lambda = 3 μm for IMG (both LM and NQ) and IMG. The minimum
RMS WFE below shall be satisfied over the full Field Of View (FOV) relevant to the
given optical path.
3. METIS-2864: The minimum Strehl ratio of the WCU+CFO+IMG-LM optical path shall be >80% at
3.3μm over the entire field of view.
4. METIS-3503: METIS shall be able to characterise the shape of the instrument PSF across the entire
FoV using the WCU.

Analysis steps:
1. Background subtraction.
2. Non-linearity correction.
3. Ramp fitting.
4. Extract the centroid position in local IMG coordinates and pixel coordinates.
5. Compare the PSF to the Zemax model PSF: FWHM + power inside first ring + contrast in the
PSF wings +PSF symmetry + radial dependence on encircled energy.

Ref. Overleaf doc IMG_OPT_02_Test_Description_PSF_Image_Quality
'''

file_path_name = '/Users/eckhartspalding/Documents/git.repos/metis_work/notebook_working_dir/data/test_IMG_OPT_02_result_1.fits'

hdul = fits.open(file_path_name)

print(hdul.info())

plt.imshow(hdul[1].data, origin='lower')
plt.show()

hdul.close()

