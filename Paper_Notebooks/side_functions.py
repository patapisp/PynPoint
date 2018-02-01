from PynPoint import Pypeline

__author__ = 'Arianna'

import math
import numpy as np
from astropy.io import fits

# ################################################################################################################################
# ################################################################################################################################


#2D gaussian fitter:
def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

################################################################################################################################
################################################################################################################################


#define a function that takes a point in the cartesian plane, takes and angle and calculate the rotated point around
#the origin point
def angular_coords_float(origin_point,start_point, degrees):
    rotation = math.radians(degrees)
    outx = origin_point[0]+(start_point[0]-origin_point[0])*math.cos(rotation) - (start_point[1]-origin_point[1])*math.sin(rotation)
    outy = origin_point[1]+(start_point[0]-origin_point[0])*math.sin(rotation) + (start_point[1]-origin_point[1])*math.cos(rotation)

    return np.array([outx,outy])

################################################################################################################################
################################################################################################################################


    ####Function that finds planets based on local maximum values
def planets_finder(image_dir,filetype,method,planet_position =[0,0],range_of_search = 0):

    ##### PARAMETER EXPLANATION #######
    ## image_dir = ARRAY or STRING = directory with the image stored as a fits file or array
    ## filetype = STRING = filetype of the input image, either 'array' or 'fits'
    ## method = STRING = method with which search for the planets, either 'global_max' (search the maximum of the entire
                        # image) or 'local_max' search for the maximum inside a region of given size and centered on
                        # planet_location
    ## planet_position = ARRAY = [x,y] = position around which search for the local maximum
    ## range_of_search = INTEGER = size of the region (centered on planet_position) inside which search for the maximum

    #Open the image depending on the filetype:
    if filetype=='array':
        image=image_dir
    if filetype=='fits':
        data = fits.open(image_dir)
        image=data[0].data
        # hdr = data[0].header

    # # Store the image dimension:
    # length_x = len(image[0])
    # length_y = len(image[1])

    #Find the maximum depending on the method input:
    if method=='local_max':
        resized_image = image[planet_position[1]-range_of_search/2.:planet_position[1]+range_of_search/2.,
                                        planet_position[0]-range_of_search/2.:planet_position[0]+range_of_search/2.]
        position = [planet_position[1]-(range_of_search/2. - np.where(resized_image==resized_image.max())[0]),
                    planet_position[0]-(range_of_search/2. -np.where(resized_image==resized_image.max())[1])]

        maximum = image[position[0][0],position[1][0]]

    if method=='global_max':
        position=np.where(image==image.max())
        maximum = image[position][0]
        # resized_image=image

    # return both the position (n.b: the np.where function returns the x and y inverted, since it actually returns the
    # position as row and column, so they need to be interchanged before returning) of the maximum and its value:

    real_position = [position[1][0],position[0][0]]

    return real_position, maximum

################################################################################################################################
################################################################################################################################
