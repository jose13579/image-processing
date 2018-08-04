#coding: utf-8

# Para rodar o codigo, primeiro, por:
#    python alinhar.py imagem_entrada.png modo_alinhamento modo_preprocesamento imagem_saida.png
# Exemplo:
#    python alinhar.py neg_4.png hough sobel neg_4_saida.png
#
# Modo alinhamento : projection e hough 
# 
# Modo preprocessamento : threshold e sobel
#			

from __future__ import unicode_literals
from scipy import ndimage
from scipy import misc
from scipy import interpolate
import numpy as np
import argparse
import sys

import skimage.filters as skif
import skimage.color as skic
import skimage.util as skiu
import skimage.transform as skim
import matplotlib.pyplot as plt
from matplotlib import cm
import pytesseract

####################### hough transformation ###################################  
def detection_hough_transformation(image):
    # Classic straight-line Hough transform
    # input: binary image
    # output: matrix, distance and angle
    acumulator, theta, rho = skim.hough_line(image)

    # The Hough transform constructs a histogram array representing the parameter space 
    #(i.e., an M×N matrix, for M different values of the distance and N different values of θ). 
    # For each parameter combination, r and θ, we then find the number of non-zero pixels in 
    # the input image that would fall close to the corresponding line, and increment the array 
    # at position (r,θ) appropriately.
    # The accumulator represent the matrix, distance and angle represent the min distance and angle
    # of the max value in the accumulator
    distance = np.where(acumulator == np.max(acumulator))[0][0] # return the distance of the higher value 
    angle = np.where(acumulator == np.max(acumulator))[1][0] # return the angle of the higher value

    # np.degrees convert angles from radians to degrees, these values are
    # in range [-90, 90]; as the rectangle rotates clockwise the
    # returned angle trends to 0, in this special case we
    # need to add 90 degrees to the angle
    # input: radians angle
    # output: degrees angle
    greater_angle = np.degrees(theta[angle])

    if greater_angle >= 45: # if greater or equal than 45 subtract 90
        greater_angle = (greater_angle - 90)

    if greater_angle <= -45: # if less or equal than 45 add 90
        greater_angle = (90 + greater_angle)

    return greater_angle, acumulator, theta, rho

####################### horizontal projection ###################################    
def detection_horizontal_projection(image):
    greater_angle = 0
    greater_value = 0
    grater_perfil = []
    valor = []
    
    # Loop from -45,to 45
    for angle in range(-45,45):

	# rotate image for each angle, using wrap interpolation
        image_rotate = skim.rotate(image, angle, resize=False, mode = "edge")
            
	# Projection of the number of black pixels in each line of the text line. 
	# Sum all pixels in the horizontal axis
        perfil = np.sum(image_rotate, axis=1)

	# Objective function, calculation of variance
	# Sum the differences of each projection on the horizontal axis and 
	# the average of all the projections on the horizontal axis raised square
        value = np.sum((perfil - np.mean(perfil))**2)
        
	# Get the max angle and projection
        if(value >= greater_value):
            greater_value = value # save the higher value
            greater_angle = angle # save angle of the higher value
	    grater_perfil = perfil # save projection of the higher value
        
    return greater_angle, grater_perfil

####################### rotate image ###################################  
def get_rotate_image(preprocess,gray,modo): 
    if(preprocess == 'sobel'):
	# Convert gray image to sobel X image
	sobel = skif.sobel_h(gray)
	# show gray and sobel images
	########################################################
	fig, axes = plt.subplots(1, 2, figsize=(15, 6))
	ax = axes.ravel()

	ax[0].imshow(gray, cmap=cm.gray)
	ax[0].set_title('Gray Image')
	ax[0].set_axis_off()

	ax[1].imshow(sobel, cmap=cm.gray)
	ax[1].set_title('Sobel X Image')
	ax[1].set_axis_off()

	plt.tight_layout()
	plt.show()
	########################################################

	preprocessed_image = sobel
    else:
	# local threshold the image, setting all foreground pixels to
	# 1 and all background pixels to 0 by block size
	block_size =17
	thresh_local = skif.threshold_local(gray, block_size, offset=10) #
	binary_local = gray > thresh_local #

	# flip the foreground and background to ensure foreground 
	# is now "white" and the background is "black"
	binary_invert = skiu.invert(binary_local)

	# show gray, threshold and invert images
	########################################################
	fig, axes = plt.subplots(1, 3, figsize=(15, 6))
	ax = axes.ravel()

	ax[0].imshow(gray, cmap=cm.gray)
	ax[0].set_title('Gray Image')
	ax[0].set_axis_off()

	ax[1].imshow(binary_local, cmap=cm.gray)
	ax[1].set_title('Threshold Image')
	ax[1].set_axis_off()

	ax[2].imshow(binary_invert, cmap=cm.gray)
	ax[2].set_title('Invert Image')
	ax[2].set_axis_off()

	plt.tight_layout()
	plt.show()
	########################################################
	
	preprocessed_image = binary_invert

    # choose hough or projection method
    if(modo == 'hough'):
	# hough method
	# input: binary image
	# output: rotation angle, acumulator hough transform, theta axis values, rho axis values
        angle_rotation, acumulator, theta, rho = detection_hough_transformation(preprocessed_image.copy())
    else:
	# horizontal projection method
	# input: binary image
	# output: rotation angle, horizontal projection values
        angle_rotation, perfil = detection_horizontal_projection(preprocessed_image.copy())
    
    # rotate the image
    # input: binary image, angle rotation
    # output: rotate image
    rotate_image = skim.rotate(gray, angle_rotation, resize=False, mode = "edge")
    
    # return rotated image
    if modo=='hough':
	return rotate_image, preprocessed_image,angle_rotation, acumulator, theta, rho
    else:
	return rotate_image, preprocessed_image,angle_rotation, perfil


# Read from console
input_img_file = sys.argv[1]
modo = sys.argv[2]
preprocess = sys.argv[3]
output_img = sys.argv[4]

####################### original image ###############################
# Read the image 
input_image = misc.imread(input_img_file)

# convert the image to grayscal
gray = skic.rgb2gray(input_image)*255

####################### rotate image ###############################
if modo=='hough':
	rotate_image, preprocessed_image, angle_rotation, acumulator, theta, rho = get_rotate_image(preprocess,gray,modo)
else:
	rotate_image, preprocessed_image, angle_rotation, perfil = get_rotate_image(preprocess,gray,modo)

####################### show images ###############################
if modo == 'hough':
    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    # show rotated image and rotation angle 
    ax[0].imshow(rotate_image, cmap=cm.gray)
    ax[0].set_title('Rotated image, rotation angle: '+str(angle_rotation))
    ax[0].set_axis_off()

    # show hough transform
    ax[1].imshow(np.log(1 + acumulator),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]],
                 cmap=cm.gray, aspect=1/1.5)
    ax[1].set_title('Accumulator Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # show detected lines of the original image
    ax[2].imshow(preprocessed_image, cmap=cm.gray)
    for _, angle, dist in zip(*skim.hough_line_peaks(acumulator, theta, rho)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - np.array(gray).shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, np.array(gray).shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, np.array(gray).shape[1]))
    ax[2].set_ylim((np.array(gray).shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    #plt.savefig("hough_"+output_img)
    plt.show()

else:
    # Generating figure 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    # show rotated image
    ax[0].imshow(rotate_image, cmap=cm.gray)
    ax[0].set_title('Rotate image')
    ax[0].set_axis_off()

    # show horizontal projection
    ax[1].plot(perfil)
    ax[1].set_title('Horizontal Projection')
    ax[1].set_xlabel('Horizontal Lines Image')
    ax[1].set_ylabel('Pixels Number')

    plt.tight_layout()
    #plt.savefig("projection_"+output_img)
    plt.show()

# show recognized text by teseract
print('text: ',pytesseract.image_to_string(rotate_image, lang='eng'))

# save image
# misc.imsave(output_img,rotate_image)


