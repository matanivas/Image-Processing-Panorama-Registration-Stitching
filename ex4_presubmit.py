import os, sys, traceback
import numpy as np
import current.sol4 as sol4
import current.sol4_utils as sol4_utils

def presubmit():
    print ('ex4 presubmission script')
    disclaimer="""
    Disclaimer
    ----------
    The purpose of this script is to make sure that your code is compliant
    with the exercise API and some of the requirements
    The script does not test the quality of your results.
    Don't assume that passing this script will guarantee that you will get
    a high grade in the exercise
    """
    print (disclaimer)
    
    print('=== Check Submission ===\n')
    if os.path.exists('current/README'):
        readme_file = 'current/README'
    elif os.path.exists('current/README.md'):
        readme_file = 'current/README.md'
    else:
        print ('No readme!')
        return False
    with open (readme_file) as f:
        lines = f.readlines()
    print ('login: ', lines[0])
    print ('submitted files:\n' + '\n'.join(map(lambda x: x.strip(), lines[1:])))
    
    print('\n=== Bonus submittes? ===')
    if os.path.exists('current/bonus.txt'):
        print ('yes, algorithm description:')
        with open('current/bonus.txt') as f:
            print (f.read())
    else:
        print ('no')
  
    print ('\n=== Section 3.1 ===\n')
    im = sol4_utils.read_image('presubmit_externals/oxford1.jpg', 1)
    try:
        im1 = im[200:300,200:400]
        im2 = im[200:300,300:500]
        print ('Harris corner detector...')
        pos = sol4.harris_corner_detector(im1)
        print ('\tPassed!')
        print ('Checking structure...')
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError('Incorrect shape of harris corner returned value.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    try:
        print ('Sample descriptor')        
        print ('Trying to build Gaussian pyramid...')
        pyr, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
        print ('\tPassed!')
        print ('Sample descriptor at the third level of the Gaussian pyramid...')
        rad  = 3
        pos = np.array([[20,25],[30,35],[40,45]])
        desc = sol4.sample_descriptor(pyr[2], pos, rad)

        print ('Checking the descriptor type and structure...')
        if desc.dtype != np.float64:
            raise ValueError('Descriptor\' type is not float64. It is %s instead.' % str(desc.dtype))
        if desc.ndim != 3 or desc.shape != (pos.shape[0], rad*2+1, rad*2+1):
            raise ValueError('Wrong shape or length of the Descriptor.')
        
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    try:
        print ('Find features.')        
        pos1, desc1 = sol4.find_features(pyr)
        if pos1.ndim != 2 or pos1.shape[1] != 2:
            raise ValueError('Incorrect shape of harris corner returned value.')
        if desc1.dtype != np.float64:
            raise ValueError('Descriptor\' type is not float64.')
        if desc1.ndim != 3 or desc1.shape != (pos1.shape[0], rad*2+1, rad*2+1):
            raise ValueError('Wrong shape or length of the Descriptor.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 3.2 ===\n')
    try:
        print ('Match Features')
#        im2 = sol4_utils.read_image('presubmit_externals/oxford2.jpg', 1)
        pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
        pos2, desc2 = sol4.find_features(pyr2)
        match_ind1, match_ind2 = sol4.match_features (desc1, desc2, .5)
        print ('\tPassed!')
        
        if match_ind1.ndim != 1 or not np.all(match_ind1.shape == match_ind2.shape):
            raise ValueError('matching indices 1 and 2 should have the same length.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 3.3 ===\n')
    try:
        print ('Compute and apply homography')
        pos1 = pos1[match_ind1,:]
        pos2 = pos2[match_ind2,:]
        H, inliers = sol4.ransac_homography(pos1, pos2, 1000, 10)

        if H.shape != (3,3):
            raise ValueError('homography should have shape (3,3)')
        if not np.isclose(H[-1,-1], 1):
            raise ValueError('homography should be normalized')
        if inliers.ndim != 1:             
            raise ValueError('inliers should have shape (S,)')

        pos1_ = sol4.apply_homography(pos1, H)
        print ('\tPassed!')
        if pos1_.ndim != 2 or pos1_.shape[1] != 2:
            raise ValueError('Incorrect shape of points after apply homography.')

        print ('display matches')
        sol4.display_matches(im1, im2, pos1, pos2, inliers)
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False


    print ('\n=== Section 3.4 ===\n')
    try:
        print ('Accumulate homographies')
        H2m = sol4.accumulate_homographies ([H], 0)
        if type(H2m) is not list:
            raise ValueError('Returned value from accumulate_homographies  should be a list!')
        if any([h.shape != (3,3) for h in H2m]):
            raise ValueError('accumulate_homographies should return a list of 3X3 homographies!')
        if len(H2m) != 2:
            raise ValueError('accumulate_homographies should return a list of length equal to the number of input images!')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False


    print ('\n=== Section 4.1 ===\n')
    try:
        print ('Warp grayscale image')
        warped_image = sol4.warp_channel(im1, H2m[0])
        if warped_image.dtype != np.float64:
            raise ValueError('warped_image is not float64. It is %s instead.' % str(warped_image.dtype))
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    try:
        print ('Compute bounding box')
        bounding_box = sol4.compute_bounding_box(H2m[0], im1.shape[1], im1.shape[0])
        if bounding_box.dtype != np.int:
            raise ValueError('bounding_box is not int. It is %s instead.' % str(bounding_box.dtype))
        if not np.allclose(bounding_box.shape,  [2,2]):
            raise ValueError('bounding_box shape is not 2x2. It is %s instead.' % str(bounding_box.shape))
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False


    print ('\n=== All tests have passed ===');
    print ('=== Pre-submission script done ===\n');
    
    print ("""
    Please go over the output and verify that there are no failures/warnings.
    Remember that this script tested only some basic technical aspects of your implementation
    It is your responsibility to make sure your results are actually correct and not only
    technically valid.""")
    return True
