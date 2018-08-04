#coding: utf-8

# Para rodar o codigo correr, por:
#    python t2.py image_file
# Exemplo:
#    python t2.py objetos1.png 

from __future__ import unicode_literals
import numpy as np
import sys
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import cv2

# Ler o arquivo da imagem e guardar isto num numpy array
img_file = sys.argv[1]
image = cv2.imread(img_file)

# Presentaçao da imagem
plt.imshow(image, cmap='gray')
plt.show()

####################### Transformação de Cores ##################################
# Usase a funçao cvtColor para transformar a imagem BGR em escadas de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Guardar e mostrar a imagem
plt.imshow(gray_image, cmap=plt.cm.gray)
plt.xlabel('Imagen em escalas de cinza')
plt.savefig('gray_object_' + img_file)
plt.show()

####################### Contornos dos Objetos ###################################
# Mascara 1
mask = [[0,-1,0],[-1,4,-1],[0,-1,0]]

# Mascara 2
mask2 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

# Filter2D realiza uma convoluçao entre a imagem em escalas de cinza
# e a mascara definida anteriormente
image_contour = cv2.filter2D(gray_image,-1,np.array(mask))

# Se invierte a imagem de entrada para obter uma nova imagem com fundo branco e bordas pretas
plt.imshow(255 - image_contour, cmap=plt.cm.gray)
plt.xlabel('Contornos dos objetos')
plt.savefig('contours_object_' + img_file)
plt.show()

###################### Extração de Propriedades dos Objetos ######################
# Se faz uso da funçao findcontours para encontrar os contornos dos objetos contidos
# na imagem filtrada
images, cnts, h = cv2.findContours(image_contour.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# se copia a imagem original e se salva em uma nova variavel
image_object = image.copy()
i = len(cnts)-1

# Para cada contorno de cada objeto encontrado na imagem de entrada (c)
# se calcula seu perimetro, area e centroide
print "Numero de regioes: ",len(cnts)
print ""
for c in cnts:
    # Area
    # Para obter a area aproximada da regiao do objeto usa-se a funçao contourArea
    # Esta funçao calcula a area do contorno do regiao do objeto
    area = cv2.contourArea(c)
    
    # Perimetro
    # Calcula o perimetro ou tamanho do arco da regiao do objeto encontrado na imagem
    perimetro = cv2.arcLength(c,True)
    # Centroide
    # Para obter o centroide se calcula os momentos de terceiro ordem
    # usando a funçao moments
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00']) #se divide o terceiro momento entre o primeiro momento para obter o centroide do objeto no direçao no eixo x
    cy = int(M['m01']/M['m00']) #se divide o segundo momento entre o primeiro momento para obter o centroide do objeto no direçao no eixo y
    
    # Se mostra cada regiao rotulada individualmente na imagem
    cv2.putText(image_object,str(i),(cx,cy), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,0,0),2)
    
    # Imprime-se a informaçao do perimetro e area da regiao de cada objeto
    print " regiao: "+(str(i))+" perimetro: "+str(perimetro)+" area: "+str(area)
    i = i-1
    
# Guardar e mostrar a imagem
plt.imshow(image_object, cmap=plt.cm.gray)
plt.xlabel('Conteo dos objetos')
plt.savefig('count_object_' + img_file)
plt.show() 

########################### Histograma de Área dos Objetos ##########################
# Se faz uso da funçao findcontours para encontrar os contornos da regiao dos objetos contidos
# na imagem filtrada
images, cnts, h = cv2.findContours(image_contour.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Para cada contorno de cada objeto encontrado na imagem de entrada (c)
# se calcula a area, se o calculo é menor a 1500 é uma regiao pequena, maior a 1500 e menor a 3000
# ou maior a 3000 é uma regiao media, e se é maior a 3000 é uma grande regiao
areas = np.array([cv2.contourArea(c) for c in cnts])           
pequeno = len(np.where(areas < 1500)[0])
medio = len(np.where( (1500 < areas) & (areas < 3000))[0])
grande = len(np.where(areas > 3000)[0])

# Imprime-se o numero de regioes por tamanho
print ""
print "numero de regioes pequenas: ",str(pequeno)
print "numero de regioes medias: ",str(medio)
print "numero de regioes grandes: ",str(grande)
print ""

########################### Histograma de Área dos Objetos ##########################
# É definido o rango das areas declaradas anteriormente
n_object = range(0,3500,1500)

# Se guarda o numero de regioes num vetor
n_area = [pequeno,medio,grande]

# Se almacena na variavel regiao o numero de objetos encontrados por numero de regioes por tamanho de area
regiao = np.arange(len(n_area))

# Se representa num grafico de barras o numero de objetos por area
plt.bar(regiao, n_area)
plt.xlabel('Area')
plt.ylabel('Nro de Objetos')
plt.xticks(regiao, n_object)
plt.title('Histograma de areas dos objetos')
plt.savefig('histograma_object_' + img_file)
plt.show()





