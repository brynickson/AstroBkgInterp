# ===================
# Authors:
#     Bryony Nickson
#     Michael Engesser
#     Justin Pierel
#     Kirsten Larson
# ===================

# Native Imports
import numpy as np
import polarTransform
# 3rd Party Imports
from scipy.signal import convolve2d
from multiprocessing import Pool
from functools import reduce

class AstroBkgInterp():
    """Astro Background Interp.

    A tool for computing reasonable background estimations for 2D or 3D
    astronomical data using a combination of interpolation and polynomial
    fitting methods to calculate the background.

    Parameters
    ----------
    src_y, src_x : int
        The (x,y) coordinates of the center of the source-masking aperture.
    bkg_mode : str
        Type of background fitting method to use on the source-masked data
        ("simple", "polynomial" or "None"). Default is "simple".
        "simple": computes a simple median for each row and column and
                  takes their weighted mean.
        "polynomial": fits 2D polynomials to small overlapping regions of
        the source-masked image and then median combines the results to
        create a smoothed background model.
        "None": No background fitting method is performed, instead the
        source-masked image is returned as the final "background".
    k : int
        Degree of the 2D polynomial fit in each dimension when `bkg_mode` =
        "polynomial". Default is 3.
    aper_rad : int
        Radius in pixels of the aperture region in which to mask the
        source(s). Default is 5.
    ann_width : int
        Width in pixels of the annulus (ring) used to estimate the
        background in the aperture region (i.e. under the source) when
        source masking. Default is 5.
    v_wht_s, h_wht_s : float
        Value by which to weight the row and column arrays when using
        `bkg_mode` is "simple". Default is 1.0, 1.0.
    kernel : ndarray
        Option to provide a 2D array of size (3,3) to use in convolving the
        masked background data. Useful for smoothing the background
        estimate. If set to None, will not convolve the masked background
        data. Default is None.
    combine_fits : bool
        Whether to combine the "polynomial" and "simple" background
        estimates. Default is False.
    mask_type : str
        Shape of the annulus aperture used to mask the source. Options are
        "circular" or "elliptical". Default is "circular".
    semi_major : int
        The length (in pixels) of the sami-major axes of the ellipse when
        `mask_type` is "elliptical". Default is 6.
    semi_minor : int
        The length (in pixels) of the semi-minor axes of the ellipse when
        `mask_type` is "elliptical". Default is 4.
    angle : int or float
        The angle (in degrees) of rotation of the ellipse with respect to
        the 2D coordinate grid. Default is 0.
    bin_size : int
        Size of the subregion(s) over which the background fitting is
        performed when `bkg_mode` is "polynomial".
    fwhm : ndarray or None
        Option to provide an input array matching the size of the cube,
        where each element is the FWHM of the PSF in that wavelength
        element.
    fwhm_scale : float
        Constant scaling factor to the PSF FWHM to get ideal aperture
        radius. Default is 1.25.
    is_cube : bool
        Whether the input data is 3D. If set to False, the data is
        assumed to be 2D. Default is False.
    err : ndarray or None
        Option to enter an error array matching the input data shape.
    uncertainties : bool
        Flag for whether an error array was passed in. Default is False.
    cube_resolution : str
        Desired resolution setting. Controls how fine the sampling is
        during the iterative polynomial fitting routine when `bkg_mode` is
        "polynomial". Options are "low", "medium", or "high".
        "low": Allows coarser sampling (larger steps, fewer fits, faster).
        "medium": Allows intermediate sampling (moderate steps and fits).
        "high": Enforces finest possible sampling (smallest steps, more
                fits, smoother background, slowest).
    """

    def __init__(self):
        # Parameters for source masking
        self.src_y, self.src_x = None, None

        self.mask_type = 'circular'

        self.aper_rad = 5
        self.ann_width = 5

        self.fwhm = None
        self.fwhm_scale = 1.25

        self.semi_major = 6
        self.semi_minor = 4
        self.angle = 0

        # Parameters for background estimation
        self.bkg_mode = 'simple'
        
        self.k = 3
        self.bin_size = 5

        self.v_wht_s = 1.0
        self.h_wht_s = 1.0

        self.kernel = None

        self.combine_fits = False

        self.is_cube = False
        self.cube_resolution = 'high'
        
        self.pool_size = 1
        
        self.err = None
        self.uncertainties = False

        return

    def print_inputs(self):
        """Print variables.

        Checks that the user has defined "src_x" and "src_y" and
        returns a print statement containing all other variables.
        """
        if self.src_y is None or self.src_x is None:
            raise ValueError("src_y and src_x must be set!")

        print(f"Source Masking: {self.mask_type}\n"
              f"    Center: {self.src_x, self.src_y}")
        if self.mask_type == 'circular':
            print(f"    Aperture radius: {self.aper_rad}")
        elif self.mask_type == 'elliptical':
            print(f"    Semi-major axis: {self.semi_major}\n"
                  f"    Semi-minor axis: {self.semi_minor}\n"
                  f"    Angle: {self.angle}")
        if self.fwhm:
            print(f"    FWHM scaling: True")
        else:
            print(f"    FWHM scaling: False")
        print(f"    Scaling factor: {self.fwhm_scale}\n"
              f"    Annulus width: {self.ann_width}\n"
              f"Background Mode: {self.bkg_mode}\n"
              f"    combine_fits: {self.combine_fits}")
        if self.bkg_mode == 'simple' or self.combine_fits:
            print(f"    v_wht_s, h_wht_s: {self.v_wht_s, self.h_wht_s}")
        elif self.bkg_mode == 'polynomial' or self.combine_fits:
            print(f"    polynomial order: {self.k}\n"
                  f"    bin size: {self.bin_size}\n"
                  f"    cube_resolution: {self.cube_resolution}\n")
        print(f"    Is cube: {self.is_cube}\n"
              f"    Convolution kernel: {self.kernel}")
        if self.pool_size != 1:
            print(f"Multiprocessing: True\n"
                 f"    pool_size: {self.pool_size}\n")
        else:
            print(f"Multiprocessing: False\n")
        print(f"Uncertainties: {self.uncertainties}\n")
            

    def get_basis(self,x, y, max_order=4):
        """Generate 2D polynomial basis functions up to a max order.

        Generates a list of polynomial basis terms of two variables (`x`,
        `y`) up to a specified maximum total degree `max_order`.

        Parameters
        ----------
        x : float or expression
            The x-variable for which the polynomials will be generated.
        y : float or expression
            The y-variable for which the polynomial will be generated.
        max_order :
            Maximum total degree of the polynomial terms. Default is 4.

        Returns
        -------
        basis
            List of polynomials terms `x^j * y^j` up to the specified
            maximum order.

        Example:
        --------
        If `x = 2`, `y = 3`, and `max_order = 2`, the result is:
        [1, x, x^2, y, x*y, x^2*y, y^2, x*y^2, x^2*y^2]

        This evaluates numberically to:
        >>> get_basis(2, 3, max_order=2)
        [1, 2, 4, 3, 6, 9]
        """
        basis = []
        for i in range(max_order+1):
            for j in range(max_order - i +1):
                basis.append(x**j * y**i)
        return basis
    

    def get_step_size(self, size, dim, resolution):
        """Calculate step size for the sliding window movement.

        Determines the step size (i.e. how far the sliding window moves)
        between each iteration of the polynomial fitting routine, based on
        the chosen `resolution` and divisibility of `dim - size`.

        Note: If `dim - size` is a prime number and resolution is *not*
        `high`, step size defaults to 1. This ensures fine-grained steps in
        challenging cases. Otherwise, the largest factor of `dim - size`
        that is within the allowed step range is chosen.

        Parameters
        ----------
        size : int
            The size of the sliding window.
        dim : int
            Dimension (height or width) of the 2D dataset.
        resolution : str
            The desired resolution setting, which controls how fine or
            coarse the step sizes should be. Must be one of `low`, `medium`,
            or `high`.

        Returns
        -------
        step : int
            The computed step size, determing how far the window moves in
            each iteration.

        Raises
        ------
        ValueError:
            If the `resolution` is not one of `low`, `medium`, or `high`.

        Example
        -------
        >>> get_step_size(size=5, dim = 20, resolution = 'low')
        2
        """
        def factors(n):
            """Return all factors of `n`."""
            return set(reduce(list.__add__, 
                        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    
        a = dim
        b = size
        c = a-b # remaining space after placing one window.

        # If c is a prime number always default to high-resolution mode.
        f = factors(c)
        if len(f) == 2 and resolution != 'high':
            print('Bin size -  Stamp size = prime number. Using a step size of 1 (high resolution).')
            step = 1
        else:
            f = sorted(f)
            # Determine maximum allowed step size.
            if resolution == 'low':
                max_step = size/2
            elif resolution == 'medium':
                max_step = size/3
            elif resolution == 'high':
                max_step = 1
            else:
                raise ValueError("Resolution must be one of 'low','medium', or 'high'.")

            reduced_f = [i for i in f if i<=max_step]
            step = max(reduced_f)

        return step


    def polyfit2d_cube(self,z,k,size):
        """Fit 2D Polynomial using sliding subregion (tiles).

        Iterates over the image in a sliding window manner, where the
        stride (step size between tiles) is determined based on the
        user-defined resolution. For each tile, the function
        fits a 2D polynomial of max order `k` to the data. The final
        background is then calculated by median combining all the fitted
        regions.

        Parameters
        ----------
        z : 2D array
            The 2D array from which to estimate the background.
        k : int
            Maximum polynomial order for the 2D fit.
        size : int
            Size of sliding window (subregion) on which the polynomial fit
            is performed.
        
        Returns
        -------
            fitted_bkg : ndarray
                The final 2D fitted background image, generated by median
                combining the fitted surfaces from all tiles.
            fitted_errs : ndarray
                The Mean Squared Error (MSE) of the fitted background array.
                Currently a placeholder set to ones.
        """
        # The total number of subregions is determined by the dimensions of
        # `z` and `size` of the sliding window.
        n = (z.shape[0]+1-size) * (z.shape[1]+1-size)

        # Initialize 3D array to store the fitted polynomial surfaces for
        # each sub-region.
        cube = np.zeros((n, z.shape[0],z.shape[1]))*np.nan
        errcube = np.zeros((n, z.shape[0],z.shape[1]))*np.nan

        count = 0

        # Compute step sizes for iterating over image
        # Vertical step size
        stepy = self.get_step_size(size,z.shape[0],self.cube_resolution)
        # Horizontal step size
        stepx = self.get_step_size(size,z.shape[1],self.cube_resolution)

        # Sliding window loop
        for j in range(size, z.shape[0]+1, stepy):
            for i in range(size, z.shape[1]+1, stepx):

                Z = z[j-size:j,i-size:i]
                x = np.arange(Z.shape[1])
                y = np.arange(Z.shape[0])

                X,Y= np.meshgrid(x,y)

                x, y = X.ravel(), Y.ravel()
                # Generate polynomial basis with maximum order
                max_order = k
                basis = self.get_basis(x, y, max_order)
                # Construct corresponding design matrix for a linear lstsq fit.
                A = np.vstack(basis).T
                b = Z.ravel()

                nans = np.isnan(b)

                # Solve polynomial coefficients using least-squares regression
                c, r, rank, s = np.linalg.lstsq(A[~nans], b[~nans], rcond=None)

                # Calculate the fitted surface from the coefficients, c.
                fit = np.sum(c[:, None, None] * np.array(self.get_basis(X, Y, max_order))
                                .reshape(len(basis), *X.shape), axis=0)
                
                # errs = np.sum(r[:, None, None] * np.array(self.get_basis(X, Y, max_order))
                #                 .reshape(len(basis), *X.shape), axis=0)

                # Reconstruct fitted surface using polynomial coefficients
                # and generated basis.
                cube[count,j-size:j,i-size:i] = fit
                # errcube[count,j-size:j,i-size:i] = errs
                
                count+=1

        # Compute median of all fitted surfaces along first axis
        fitted_bkg = np.nanmedian(cube,axis=0)
        fitted_errs = np.ones_like(fitted_bkg)#np.nanmean(errcube,axis=0) # Mean Squared Error
        
        return fitted_bkg, fitted_errs


    def interpolate_source(self, data, center, is_err=False):
        """Interpolate the sky under the source.

        Uses linear interpolation to mask the source in the aperture region
        using the median background in a user-defined annulus.

        Parameters:
        -----------
        data: ndarray
            The 2D input data in Cartesian coordinates.

        center: Tuple[float, float]
            The (x, y) coordinates of the source, used as the center for
            the polar tansformation.
        is_err: bool
            Whether data is an error array. Default is False.

        Returns:
        --------
        cartesianImage: ndarray
            The final background-interpolated image in cartesian
            coordinates.
            The background-interpolated image in cartesian coordinates.
            The filtered and interpolated image data in cartesian
            coordinates.

        Notes
        -----
        - The background values are interpolated radially and symmetrically.
        - Handles uncertainty propagation when needed.
        - Circular interpolation uses radial symmetry
        - Elliptical interpolation uses elliptical symmetry with orientation.
        """
        # Ensure original data is not modified.
        cartesian_data = data.copy()
        # Convert from Cartesian to Polar coordinates, centered on source.
        polarImage, ptSettings = polarTransform.convertToPolarImage(
            cartesian_data, center=center)

        m, n = polarImage.shape
        temp = np.zeros(m) # For storing median background values.
        half = m // 2  # Mid-point index for symmetry in circular interpolation
        mask_type = self.mask_type
        r = self.aper_rad
        ann_width = self.ann_width

        if mask_type == 'circular':
            # Compute median background for each row (angular slice) in
            # annulus region
            for i in range(0, m):
                if is_err:
                    # compute uncertainties of annulus in quadrature
                    temp[i] = np.sqrt(np.sum([polarImage[i,r+j]**2 for j in range(ann_width)]))
                else:
                    temp[i] = np.nanmedian(polarImage[i, r:r + ann_width])

            # Interpolate background radially into aperture region using
            # linear interpolation between opposite rows

            # linear interpolation between the median values of the opposite rows in the polar image.

            # Interpolate background values symmetrically into aperture region.
            for i in range(0, m):
                if is_err:
                    # add uncertainties of aperture edges in quad because
                    # there is a subtraction in this step
                    sigma = np.sqrt(temp[i]**2 + temp[i - half]**2)
                    new_row = np.ones(r*2)*sigma
                else:
                    new_row = np.linspace(temp[i], temp[i - half], r * 2)

                # fill in the left and right halves of the row in the polar image with the interpolated values.
                polarImage[i, :r] = new_row[r - 1::-1] # left half (mirrored)
                polarImage[i - half, :r] = new_row[r:] # right half

        elif mask_type == 'elliptical':
            a, b = self.semi_major, self.semi_minor
            an, bn = a + ann_width, b + ann_width  # annulus semi-major and semi-minor axes
            angle = self.angle

            # define a theta array (angular positions around the ellipse)
            # Compute angle-dependent radial profile
            n = int(angle / 360 * m)
            theta1 = np.linspace(np.deg2rad(angle), 2 * np.pi, m - n)
            theta2 = np.linspace(np.deg2rad(angle), np.deg2rad(angle), n)
            theta = np.concatenate([theta1, theta2], axis=0)

            # Equation for radius of an ellipse as a function of a, b, and theta
            # Compute ellipse aperture and annulus for each angle
            rap = (a * b) / np.sqrt((a ** 2) * np.sin(theta) ** 2 + (b ** 2) * np.cos(theta) ** 2)
            rann = (an * bn) / np.sqrt((an ** 2) * np.sin(theta) ** 2 + (bn ** 2) * np.cos(theta) ** 2)

            # Interpolate background into ellptical aperture region
            for i in range(polarImage.shape[0]):
                if is_err:
                    # propogate uncertainty
                    polarImage[i, :int(rap[i])] = np.sqrt(np.sum([polarImage[i, int(rap[i])+j]**2 for j in range(int(rann[i]))]))

                else:
                    # assign median background inside aperture
                    polarImage[i, :int(rap[i])] = np.nanmedian(polarImage[i, int(rap[i]):int(rann[i])])

        # Convert modified polar image back Cartesian coordinates
        cartesianImage = ptSettings.convertToCartesianImage(polarImage)

        return cartesianImage

    def mask_source(self, data, is_err=False):
        """Mask the source using interpolated background.

        Masks the source in the input data by interpolating the background
        under the source at several dithered positions and computing the
        median of the resulting images. If a convolution kernel is provided,
        the background estimate is convolved with it before being returned.

        Parameters
        ----------
        data : ndarray
            The 2D input data to be masked.
        is_err: bool
            Whether data is an error array. Default is False.

        Returns
        -------
        conv_bkg : ndarray
            The 2D median combined source-masked image (after convolution,
            if kernel is provided).
        """
        dithers = []  # for storing dithered images

        # iterate over neighboring pixels in a 3x3 grid around the source.
        for i in range(-1, 2):
            for j in range(-1, 2):
                center = [self.src_x + i, self.src_y + j]  # coordinates of shifted copy
                dither = self.interpolate_source(data, center, is_err=is_err)  # interpolate shifted copy
                dithers.append(dither)

        # Compute median of the dithered images
        if is_err:
            # add uncertainties in quadrature
            dithers2 = np.array(dithers)**2
            new_data = np.sqrt(np.sum(dithers2,axis=0))
        else:
            new_data = np.nanmedian(np.array(dithers), axis=0)

        # convolve with kernel if one was passed in
        if self.kernel is not None:
            conv_bkg = convolve2d(new_data, self.kernel, mode='same',boundary='symm')
        else:
            # If no kernel is provided, use median combined data directly
            conv_bkg = new_data

        return conv_bkg

    def normalize_poly(self, bkg_poly, bkg_simple):
        """Normalize and blend polynomial fit with median fit.

        Smooths out the polynomial background using the median background
        to create a combined normalized background image. The method
        involves first normalizing both background images, multiplying them
        element-wise and then rescaling back to the original polynomial
        background range.

        Parameters:
        -----------
        bkg_poly : ndarray
            The 2D polynomial background image.
        bkg_simple : ndarray
            The 2D simple background image.

        Returns:
        --------
        combo : ndarray
            The 2D combined normalized background image.
        """
        # Compute min and max for both backgrounds
        polymax = np.nanmax(bkg_poly)
        polymin = np.nanmin(bkg_poly)
        simplemax = np.nanmax(bkg_simple)
        simplemin = np.nanmin(bkg_simple)

        # Normalize the backgrounds based on their max and min values
        norm1 = (bkg_poly - polymin) / (polymax- polymin)
        norm2 = (bkg_simple - simplemin) / (simplemax - simplemin)

        # combine the normalized polynomial and simple backgrounds and
        # scale by the polynomial maxmimum
        combo = (norm1 * norm2)
        combo *= (polymax-polymin)
        combo += simplemin

        return combo

    def simple_median_bkg(self, data, v_wht=1., h_wht=1.):
        """Calculate simple median background estimate.

        Computes a simple background estimate for the masked input data by
        calculating the weighted average of the row and column medians.

        Parameters
        ----------
        data : ndarray
            The 2D source-masked data from which the background will be
            estimated.
        v_wht : float
            Weight to apply to the per-column median background. Default is
            1.0.
        h_wht : float
            Weight to apply to the per-row median background. Default is
            1.0.

        Returns
        -------
        bkg : ndarray
            The 2D simple background estimate.
        """
        bkg_h = np.zeros_like(data)
        bkg_v = np.zeros_like(data)
        m, n = data.shape

        # Compute median of each row
        for i in range(0, m):
            bkg_h[i, :] = np.nanmedian(data[i, :], axis=0)

        # Compute median of each column
        for j in range(0, n):
            bkg_v[:, j] = np.nanmedian(data[:, j], axis=0)

        # Calculate weighted average of the row and column medians
        bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht, h_wht], axis=0)
        bkg = bkg_avg

        return bkg

    def interp_nans(self, data):
        """Interpolate NaN pixels using neighboring pixels.

        Replaces NaN values in input data with a median of their
        neighboring values in a 3x3 window centered around the NaN. If all
        neighbors are NaN, the value is set to 0. Edge cases are handled to
        ensure window does not go out of bounds.
        
        Parameters:
        -----------
        data : 2D numpy array
            Input data containing NaNs.
        
        Returns:
        -----------
        newdata : 2D numpy array
            Modified copy of the input data where NaNs have been replaced
            by either their local median or zero, where no valid neighbors
            exist.
        """
        m,n = data.shape
        newdata = np.copy(data)

        for j in range(m):
            for i in range(n):
                if np.isnan(data[j,i]):
                    # Define bounds to avoid index errors
                    j_min, j_max = max(j - 1, 0), min(j + 2, m)
                    i_min, i_max = max(i - 1, 0), min(i + 2, n)

                    # Compute median of neighbouring pixels, ignoring NaNs
                    med = np.nanmedian(data[j_min:j_max, i_min:i_max])

                    # Assign median, else zero if no valid neighbors
                    if np.isnan(med):
                        newdata[j,i] = 0
                    else:   
                        newdata[j,i] = med

        return newdata
    
    def process(self,i):
        """Process slice to estimate and subtract the background.

        Processes a single image or slice of a 3D image cube by applying
        source masking, estimating the background depending on the selected
        mode, and optionally propagates errors.

        Parameters
        ----------
        i : int
            Index of the slice to process, if the data is a 3D cube.

        Returns
        -------
        diff : ndarray
            The background-subtracted image.
        bkg : ndarray
            The estimated background.
        masked_bkg : ndarray
            The source masked image.

        """
        # Extract and process single 2D image
        if not self.is_cube:
            im = self.data
            im = self.interp_nans(im[0])
            if self.uncertainties:
                err = self.err
        else:
            im = self.data[int(i)].copy() # Extract the i-th slice
            im = self.interp_nans(im)
            if self.uncertainties:
                err =self.err[int(i)].copy()

        if self.fwhm is not None:
            self.aper_rad = int(np.ceil(self.fwhm_scale * self.fwhm[i]))

        if self.uncertainties:
            masked_err = self.mask_source(err, is_err=True)
            masked_err = np.array([masked_err])

        masked_bkg = self.mask_source(im)
        masked_bkg = np.array([masked_bkg])

        if self.bkg_mode == 'polynomial':
            bkg, res = self.polyfit2d_cube(masked_bkg[0],self.k,self.bin_size)
            
            if self.uncertainties:
                bkg_err = np.sqrt(masked_err**2+res**2)

            if self.combine_fits:
                bkg_simple = self.simple_median_bkg(masked_bkg[0], v_wht=self.v_wht_s, h_wht=self.h_wht_s)
                bkg = self.normalize_poly(bkg, bkg_simple)

        elif self.bkg_mode == 'simple':
            bkg = self.simple_median_bkg(masked_bkg[0], v_wht=self.v_wht_s, h_wht=self.h_wht_s)

        else:
            bkg = masked_bkg[0]

        diff = im - bkg

        if self.uncertainties:
            diff_err = np.sqrt(err**2 + bkg_err**2)
        else:
            diff_err = None
            
        return diff, bkg, masked_bkg#, diff_err#, masked_err, bkg_err

    def run(self, data, err=None):
        """Run background subtraction on 2D or 3D image data.

        Runs the background estimation and subtraction on the input data,
        optionally handling error propagation.

        Parameters
        ----------
        data: ndarray
            The 2D or 3D input data.
        err : ndarray, optional
            The 2D or 3D uncertainty array associated with the data.

        Returns
        -------
        diff: ndarray
            The background-subtracted data.
        bkg: ndarray
            The estimated background.
        masked_bkg: ndarray
            The source-masked data.
        """
        if err is not None:
            self.err = err
            self.uncertainties = True

        # Print Inputs
        self.print_inputs()

        ndims = len(data.shape)

        # Check if data is 3D cube or 2D image
        if ndims == 3:
            k = data.shape[0]
            masked_bkgs = np.zeros_like(data)
            bkgs = np.zeros_like(data)
            diffs = np.zeros_like(data)
            errs = np.zeros_like(err)
            self.is_cube = True
        else:
            # Convert 2D data to 3D format
            k = 1
            data = np.array([data])
            err = np.array([err])

        self.data = data
        if k == 1:
            diffs, bkgs, masks, errs = self.process(0)#merrs, berrs
        else:
            # Process multiple slices in parallel using multiprocessing
            p = Pool(self.pool_size)
            idx = np.arange(k)
            # diffs, bkgs, masks = p.map(self.process,idx)
            results = p.map(self.process,idx)
            diffs,bkgs,masks = zip(*results)
            # results = np.array(results)

            # diffs = results[:,0]
            # bkgs = results[:,1]
            # masks = results[:,2]
            #errs = results[:,3]
            # merrs = results[:,4]
            # berrs = results[:,5]

        masked_bkg = np.array(masks)
        bkg = np.array(bkgs)
        diff = np.array(diffs)
        # err = np.array(errs)
        # merr = np.array(merrs)
        # berr = np.array(berrs)
        
        # If original input was 2D, return 2D outputs
        if not self.is_cube:
            masked_bkg = masked_bkg[0]
            err = err[0]
            # merr = merr[0]
            # berr = berr[0]

        return diff, bkg, masked_bkg#, err#,merr,berr
