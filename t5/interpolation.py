
# coding: utf-8

# # Trabalho 4 Processamento de Imagens
# python interpolation.py [--a,--e,--d] [angulo, escala, dimensao_x dimensao_y] --m tipo_interpolaçao --i imagem_entrada.png --o imagem_saida.png

# python interpolation.py --a 45 --m bilinear --i imagem1.png --o imagem1_saida.png 
# para rotar uma imagem 45 graus

# python interpolation.py --e 2. --m bicubic --i imagem1.png --o imagem1_saida.png 
# para escalar uma imagem por 2

# python interpolation.py --d 150 100 --m nearest --i imagem1.png --o imagem1_saida.png 
# para redimensionar uma imagem

# tipo_interpolaçao : nearest, bilinear, bicubic ou lagrange

# ## Declaraçao das bibliotecas
from __future__ import division
from scipy import misc
import numpy as np
import argparse
import sys
import math
import skimage.color as skic
import matplotlib.pyplot as plt
from matplotlib import cm
#import argparse


# ## Interpolação
# ### Interpolação do vizinho mais perto
def Nearest_Interpolation(image,indx_imagex,indx_imagey,(new_h , new_w)):
    print "nearest"
    # Calculo da altura e largura
    h, w = image.shape
    
    # Criar uma nova matriz de zeros (imagem reescalada ou rotada)
    output_img = np.zeros((new_h, new_w))

    # Redondeo das posiçoes em X e Y de cada pixel da imagem reamostrada representados 
    # na imagem de entrada 
    # round é uma função que aproxima um número para seu valor inteiro mais próximo.
    x_round = np.round(indx_imagex).astype(int)
    y_round = np.round(indx_imagey).astype(int)

    for i in range(0,new_h):
        for j in range(0,new_w):
            # Para cada posiçao redondeada em X e Y
            nx_round = x_round[i][j]            
            ny_round = y_round[i][j]  

            # Se a posiçao redondeada em X é maior a 0 e menor a altura da imagem de entrada
            # e se a posiçao redondeada em X é maior a 0 e menor a largura da imagem de entrada
            # Escrever o valor da intensidade de cada pixel na posiçao "nx_round,ny_round" da 
            # imagem de entrada, para o pixel a ser atribuido na posiçao i,j na imagem reamostrada
            if nx_round >= 0 and nx_round < h and ny_round >= 0 and ny_round < w: 
                output_img[i,j] = image[nx_round,ny_round]   
                        
    return output_img


# ### Interpolaçao Bilinear
def Bilinear_Interpolation(image,indx_imagex,indx_imagey,(new_h , new_w)):
    print "bilinear"
    # Calculo da altura e largura
    h, w = image.shape
    
    # Criar uma nova matriz de zeros (imagem reamostrada)
    output_img = np.zeros((new_h, new_w))
     
    # Calculo das novas posiçoes dos quatro vizinhos mais proximos
    x = np.floor(indx_imagex).astype(int)
    y = np.floor(indx_imagey).astype(int)
    
    x_1 = np.floor(indx_imagex).astype(int)+1
    y_1 = np.floor(indx_imagey).astype(int)+1

    # Calculo das distancias nas direçoes X e Y
    # diffX = X´ - X
    # diffY = Y´ - Y
    dx = indx_imagex - np.floor(indx_imagex)
    dy = indx_imagey - np.floor(indx_imagey)

    for i in range(0,new_h):
        for j in range(0,new_w):
            nx = x[i,j]
            ny = y[i,j]
            nx_1 = x_1[i,j]
            ny_1 = y_1[i,j]
  
            # Se as posiçoes em X dos quatro pixels vizinhos mais proximos sao maiores a 0
            # e menores à altura da imagem de entrada e se as posiçoes em Y dos quatro pixels 
            # vizinhos mais proximos sao maiores a 0 e menores à largura da imagem de entrada
            if nx_1 >= 0 and nx_1 < h and ny_1 >= 0 and ny_1 < w:
                # Calculo da meia ponderada de distancia dis quatro pixeis vizinhos mais proximos
                # para determinar a intensidade de cada pixel na imagem amostrada
                top = (1-dx[i,j])*image[nx,ny]+(dx[i,j])*image[nx_1,ny]
                bottom = (1-dx[i,j])*image[nx,ny_1]+(dx[i,j])*image[nx_1,ny_1]

                output_img[i,j] = (1-dy[i,j])*top+(dy[i,j])*bottom
                    
    return output_img


# ### Interpolaçao Bicubica
def P(t):
    return np.where(t>0,t,0)
    
def R(s):
    return (math.pow(P(s + 2),3) - 4 * math.pow(P(s + 1),3) + 6 * math.pow(P(s),3) - 4 * math.pow(P(s - 1),3))/6.0
    
def Bicubic_Interpolation(image,indx_imagex,indx_imagey,(new_h , new_w)):
    print "bicubic"
    # Calculo da altura e largura
    h, w = image.shape
    
    # Criar uma nova matriz de zeros (imagem reamostrada)
    output_img = np.zeros((new_h, new_w))
     
    # Redondear para abaixo as posiçoes mapeadas em X e Y de cada pixel da 
    # imagem amostrada na imagem de entrada
    x = np.floor(indx_imagex).astype(int)
    y = np.floor(indx_imagey).astype(int)

    # Calculo das distancias nas direçoes X e Y
    # diffX = X´ - X
    # diffY = Y´ - Y
    diffx = indx_imagex - np.floor(indx_imagex)
    diffy = indx_imagey - np.floor(indx_imagey)
    
    for i in range(0,new_h):
        for j in range(0,new_w):
            new_pixel = 0
            # Calculo da vizinhança de 4 x 4 pixeis ao redor do pixel em questão
            # para calcular seu valor de intensidade
            for m in range(-1,3):
                for n in range(-1,3):
                    # Calculo das posiçoes em X e Y para cada vizinho 4x4 ao redor 
                    # de cada pixel na imagem de entrada
                    nx = x [i,j] + m
                    ny = y[i,j] + n

                    # Distancia entre cada vizinho 4x4 ao redor 
                    # de cada pixel na imagem de entrada 
                    ndiffx = m - diffx[i,j]
                    ndiffy = diffy[i,j] - n

                    # Se as posiçoes para cada vizinho em X e Y sao maiores a 0 e menores à
                    # altura da imagem de entrada e se sao maiores a 0 e menores à largura
                    # da imagem de entrada
                    if nx >= 0 and nx < h and ny >= 0 and ny < w:
                        #Calculo da funçao B-spline cúbica
                        new_pixel += image[nx,ny] * R(ndiffx) * R(ndiffy)

            output_img[i,j] = new_pixel

    return output_img
                            


# ### Interpolaçao de Lagrange
def L(n,dx,image,x,y,h,w):
    if x - 1 >= 0 and x + 1 < h and x + 2 < h and y + n - 2 >= 0 and y + n - 2 < w:
        return (-1.0 *dx*(dx-1)*(dx-2)*image[x-1,y+n-2])/6.0 + ((dx+1)*(dx-1)*(dx-2)*image[x,y+n-2])/2.0 +              (-1.0 * dx*(dx+1)*(dx-2)*image[x+1,y+n-2])/2.0 + (dx*(dx+1)*(dx-1)*image[x+2,y+n-2])/6.0
    else:
        return 0.0
                    

def Lagrange_Interpolation(image,indx_imagex,indx_imagey,(new_h , new_w)):
    print "lagrange"
    # Calculo da altura e largura
    h, w = image.shape
    
    # Criar uma nova matriz de zeros (imagem reamostrada)
    output_img = np.zeros((new_h, new_w))
     
    # Rendondear para abaixo as posiçoes mapeadas em X e Y da 
    # imagem amostrada na imagem de entrada
    x = np.floor(indx_imagex).astype(int)
    y = np.floor(indx_imagey).astype(int)

    # Calculo das distancias nas direçoes X e Y
    # diffX = X´ - X
    # diffY = Y´ - Y
    dx = indx_imagex - np.floor(indx_imagex)
    dy = indx_imagey - np.floor(indx_imagey)
    
    for i in range(0,new_h):
        for j in range(0,new_w):
            # Calculo do valor de intensidade de cada pixel para a imagem amostrada
            # utilizando polinomios de Lagrange
            output_img[i,j] = (-1.0*dy[i,j]*(dy[i,j]-1)*(dy[i,j]-2)*L(1,dx[i,j],image,x[i,j],y[i,j],h,w))/6.0 + ((dy[i,j]+1)*(dy[i,j]-1)*(dy[i,j]-2)*L(2,dx[i,j],image,x[i,j],y[i,j],h,w))/2.0 +  (-1.0*dy[i,j]*(dy[i,j]+1)*(dy[i,j]-2)*L(3,dx[i,j],image,x[i,j],y[i,j],h,w))/2.0 + (dy[i,j]*(dy[i,j]+1)*(dy[i,j]-1)*L(4,dx[i,j],image,x[i,j],y[i,j],h,w))/6.0

    return output_img


# ## Operaçoes
# ### Escalamento
def image_rescale(image,scale_factor,interpolation_type):
    # Calculo da altura e largura
    h, w = image.shape
    
    # Calculo da nova altura e largura para a imagem reescalada
    new_h = h * scale_factor # altura da imagem reescalada
    new_w = w * scale_factor # largura da imagem reescalada

    # Logo do calculo da nova altura e largura é redondeado e
    # Convertido a inteiro
    round_new_h = int(np.round(new_h))
    round_new_w = int(np.round(new_w))

    # Cria-se uma matriz de indices com o tamanho da nova imagem reescalada
    # Esta matriz representa as posiçoes em X e Y de cada pixel da imagem reescalada 
    indx_rescale_image = np.indices((round_new_h,round_new_w))

    # As posiçoes em X e Y de cada pixel da imagem reescalada são divididos pelo fator de escala
    # Esta divisao mapea cada posiçao de cada pixel da imagem reescalada para
    # cada posiçao correspondente na imagem de entrada
    indx_rescale_imagex = indx_rescale_image[0] / scale_factor
    indx_rescale_imagey = indx_rescale_image[1] / scale_factor
    
    # Para determinar o valor do pixel de cada posiçao correspondente na imagem de 
    # entrada, são utilizados diferentes metodos de interpolaçao
    if interpolation_type == "nearest":
        interp_image = Nearest_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(round_new_h, round_new_w))
    else:
        if interpolation_type == "bilinear":
            interp_image = Bilinear_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(round_new_h, round_new_w))
        else:
            if interpolation_type == "bicubic":
                interp_image = Bicubic_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(round_new_h, round_new_w))
            else:
                interp_image = Lagrange_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(round_new_h, round_new_w))

    return interp_image


# ### Rotação
def image_rotate(image,angle,interpolation_type):
    # Calculo da altura e largura
    h, w = image.shape
    
    # Calculo do centroide da imagem de entrada
    centerx,centery = (h / 2, w / 2)
    
    # Restabelecer o valor do angulo para negativos
    angle = -1*angle
    
    # Transformaçao a radianes do angulo de entrada
    angle = math.radians(angle)
    
    # calculo do seno do angulo em radianes
    # Calculo do coseno do angulo em radianes
    sin = math.sin(angle)
    cos = math.cos(angle)
    
    # Cria-se uma matriz de indices com o tamanho da nova imagem rotada
    # Esta matriz representa as posiçoes em X e Y de cada pixel da imagem rotada 
    indx_rescale_image = np.indices((h,w))
    
    # Subtraia-se o valor do centro da imagem a cada posição em X e Y de cada pixel da imagem rotada 
    # Esta operação desloca as posiçoes originais da imagem de entrada ao centro no plano X Y, onde as novas 
    # posiçoes do centro da imagem de entrada seja (0,0)
    indx_rescale_center_imagex = indx_rescale_image[0]-centerx
    indx_rescale_center_imagey = indx_rescale_image[1]-centery
    
    # Estas operaçoes mapeam cada posiçao de cada pixel em X e Y da imagem rotada para
    # cada posiçao em X e Y correspondente na imagem de entrada 
    # x_rotada = x cos(theta) - y sin(theta)
    # y_rotada = x sin(theta) + y cos(theta)
    indx_rescale_imagex = indx_rescale_center_imagex*cos - indx_rescale_center_imagey*sin
    indx_rescale_imagey = indx_rescale_center_imagex*sin + indx_rescale_center_imagey*cos
    
    # Soma-se o valor do centro da imagem a cada posiçao em X e Y de cada pixel da imagem rotada
    # Esta operaçao desloca a imagem de entrada no centro a suas posiçoes originais no plano X Y
    indx_rescale_imagex += centerx
    indx_rescale_imagey += centery
    
    # Para determinar o valor do pixel de cada posiçao correspondente na imagem de 
    # entrada, são utilizados diferentes metodos de interpolaçao
    if interpolation_type == "nearest":
        interp_image = Nearest_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(h, w))
    else:
        if interpolation_type == "bilinear":
            interp_image = Bilinear_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(h, w))
        else:
            if interpolation_type == "bicubic":
                interp_image = Bicubic_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(h, w))
            else:
                interp_image = Lagrange_Interpolation(image.copy(),indx_rescale_imagex,indx_rescale_imagey,(h, w))

    return interp_image

# ### Redimensão
def image_resize(image,(new_h,new_w),interpolation_type):
    # Calculo da altura e largura
    h, w = image.shape
    
    # Cria-se uma matriz de indices com o tamanho da nova imagem redimensionada
    # Esta matriz representa as posiçoes em X e Y de cada pixel da imagem redimensionada 
    indx_redimensionar_image = np.indices((new_h,new_w))
    
    # Calcular o fator de escala em X e Y para a imagem redimensionada
    scale_factorx = new_h / h
    scale_factory = new_w / w
    
    # As posiçoes em X e Y de cada pixel da imagem redimensionada são divididos pelo fator de escala
    # Esta divisao mapea cada posiçao de cada pixel da imagem redimensionada para
    # cada posiçao correspondente na imagem de entrada
    indx_redimensionar_imagex = indx_redimensionar_image[0] / scale_factorx
    indx_redimensionar_imagey = indx_redimensionar_image[1] / scale_factory

    if interpolation_type == "nearest":
        interp_image = Nearest_Interpolation(image.copy(),indx_redimensionar_imagex,indx_redimensionar_imagey,(new_h, new_w))
    else:
        if interpolation_type == "bilinear":
            interp_image = Bilinear_Interpolation(image.copy(),indx_redimensionar_imagex,indx_redimensionar_imagey,(new_h, new_w))
        else:
            if interpolation_type == "bicubic":
                interp_image = Bicubic_Interpolation(image.copy(),indx_redimensionar_imagex,indx_redimensionar_imagey,(new_h, new_w))
            else:
                interp_image = Lagrange_Interpolation(image.copy(),indx_redimensionar_imagex,indx_redimensionar_imagey,(new_h, new_w))

    return interp_image

# print images
def print_images(gray_image, interpolated_image,inter_type, output_image):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
	ax = axes.ravel()

	ax[0].imshow(gray_image, cmap=cm.gray)
	ax[0].set_title('Original Image')
	ax[0].set_axis_off()

	ax[1].imshow(interpolated_image, cmap=cm.gray)
	ax[1].set_title(inter_type+" Interpolation")
	ax[1].set_axis_off()

	plt.tight_layout()
	plt.savefig(output_image)
	plt.show()

# Run program 
def run(gray_image,action,parameter1,parameter2,interpolation_type,output_image):
	if action == "--a":
		# ### Rotaçao
		# Chama-se as funçoes por tipo de interpolaçao para rotar uma imagem
		angle = float(parameter1)
		interpolated_image = image_rotate(gray_image.copy(),angle,interpolation_type)
 		print_images(gray_image,interpolated_image,interpolation_type,output_image)
	else:
		if action == "--e":
			scale = float(parameter1)
			if scale > 0.:
				# ### Scaling
				# Chama-se as funçoes por tipo de interpolaçao para escalar uma imagem
				interpolated_image = image_rescale(gray_image.copy(),scale,interpolation_type)
				print_images(gray_image,interpolated_image,interpolation_type,output_image)
			else:
				print "Erro, o fator de escala não pode ser menor a zero"
		else:
			dimensionx = int(parameter1)
			dimensiony = int(parameter2)
			if dimensionx > 0 and dimensiony > 0:
				# ### resizing
				# Chama-se as funçoes por tipo de interpolaçao para redimensionar uma imagem
				interpolated_image = image_resize(gray_image.copy(),(dimensionx,dimensiony),interpolation_type)
				print_images(gray_image,interpolated_image,interpolation_type,output_image)
			else:
				print "Erro, a novas dimençoes não podem ser menor ou igual a zero"


# ## Desarrollo do problema
# ### Declaraçao das variaveis globais
input_file = sys.argv[0]
action = sys.argv[1]
parameter1 = sys.argv[2]
parameter2 = 0.

if action == "--d":
	parameter2 = sys.argv[3]
	action2 = sys.argv[4]
	interpolation_type = sys.argv[5]
	action3 = sys.argv[6]
	input_img_file = sys.argv[7]
	action4 = sys.argv[8]
	output_img_file = sys.argv[9]
else:
	action2 = sys.argv[3]
	interpolation_type = sys.argv[4]
	action3 = sys.argv[5]
	input_img_file = sys.argv[6]
	action4 = sys.argv[7]
	output_img_file = sys.argv[8]

try:
	assert action in ["--a","--e","--d"]
except AssertionError:
	print 'Erro, a entrada deve ser "--a","--e" ou "--d": ' + action
	raise

try:
	assert action2 in ["--m"]
except AssertionError:
	print 'Erro, a entrada deve ser "--m": ' + action2
	raise

try:
	assert action3 in ["--i"]
except AssertionError:
	print 'Erro, a entrada deve ser "--i": ' + action3
	raise

try:
	assert action4 in ["--o"]
except AssertionError:
	print 'Erro, a entrada deve ser "--o": ' + action4
	raise

try:
	assert interpolation_type in ["nearest","bilinear","bicubic","lagrange"]
except AssertionError:
	print 'Erro, a entrada deve ser "nearest","bilinear","bicubic" ou "lagrange": ' + interpolation_type
	raise


# ### Ler imagem de entrada
input_image = misc.imread(input_img_file)


# ### Gray scale
# Transforma uma imagem colorida a uma imagem em escala de cinza
gray_image = skic.rgb2gray(input_image)*255

# ### Run program
run(gray_image.copy(),action,parameter1,parameter2,interpolation_type,output_img_file)

