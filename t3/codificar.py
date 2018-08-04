#coding: utf-8

# Para rodar o codigo correr, por:
#    python codificar.py image_file_input.png input_text plane_number image_file_output.png
# Exemplo:
#    python codificar.py image1.png texto_entrada.txt 0 image1_saida.png

from __future__ import unicode_literals
import numpy as np
import sys
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import csv
from math import log

# Obter uma componente da imagem
# entrada: imagem de entrada, numero da componente
def get_component(image,nro_component):
    return image[:,:,nro_component]

# Obter a messagem oculta em codigo binario 
# entrada: palabvra
def get_cod_bin_word(word):
    merge_bin = ""
    for character in word:
        character_ascii = ord(character) # transformar caracter para codigo ASCII
        ascii_bin = bin(character_ascii) # transformar codigo ASCII para codigo binario
        merge_bin += ascii_bin[2:].zfill(8) # tirar os dois bits caracteres e cheiar até que o codigo binario tenha tamanho oito

    #retornar codigo binario da palavra
    return merge_bin


# Ler a palavra do arquivo de texto
# entrada: nome do arquivo de entrada
def read_word(input_text):
    #read it
    with open(input_text, 'r') as csvfile:
        reader = csv.reader(csvfile)
        word = [[e for e in r] for r in reader]
    return np.array(word)[0][0]

# Metodo que codifica a messagem no bit menos significativo
# entrada: imagem, plano de bits e texto entrada
# saida: imagem com messagem codificada
def get_word_encoded_image(image, plane_number, input_text,mask):
    indx = 0
    flag = False

    # ler a palavra do arquivo de entrada
    # concatenar um criterio de parada para nossa mensagem
    word = read_word(input_text)+"]"

    # transformar a mensagem a codigo binario 
    bin_word = get_cod_bin_word(word)
    
    # para cada pixel de cada componente da imagem
    for w in range(image.shape[0]):
        for h in range(image.shape[1]):   
            for c in range(3):

		# ler cada bit da mensagem como um inteiro
                bin_value = int(bin_word[indx])

		# Para cada pixel de cada componente da imagem, fazer uma operaçao logica "and" ou "or"  
		# com uma mascara de bits para mudar o bit menos significativo 
                bit_pixel_mudado_zero = image[w,h,c] & (255 - mask[plane_number])
                bit_pixel_mudado_um = image[w,h,c] | mask[plane_number]

		# se o bit menos significativo é zero se aplica uma operaçao logica "and"
		# caso contrario se aplica uma operaçao "or" ao pixel, fazendo uso de mascaras de bits
                image[w,h,c] = np.where(bin_value == 0,bit_pixel_mudado_zero,bit_pixel_mudado_um)
  
		# aumentar o contador
                indx= indx + 1

		# se o contador é igual ou maior ao tamanho da palavra muda o flag para true e 
		# é qebrado o loop 
                if(indx >= len(bin_word)):
                    flag = True
                    break
                    
            if(flag==True): 
                break
                
        if(flag==True): 
            break

    # retornar imagem com mensagem oculta  
    return image




input_img_file = sys.argv[1]
input_text = sys.argv[2]
plane_number = int(sys.argv[3])
output_img_file = sys.argv[4]
mask = [1,2,4,8,16,32,64,128]

# Ler o arquivo da imagem
image = misc.imread(input_img_file)

################################ Codificar palavra ###################################
# Encoded word in the image
image_encoded = get_word_encoded_image(image.copy(), plane_number,input_text,mask)


####################### mostrar imagem sem texto codificado ###############################
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Imagem sem mensagem oculta')
plt.axis('off')

fig, big_axes = plt.subplots( figsize=(15.0, 15.0) , nrows=3, ncols=1, sharey=True) 

for row, big_ax in enumerate(big_axes, start=1):
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    big_ax._frameon = False

idx = 1
for component in range(1,4):
    image_component = get_component(image.copy(),component-1)
    
    ax = fig.add_subplot(3,5,idx)
    ax.imshow(image_component, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_title('Canal ' +str(component)+' da imagem ')
    idx = idx + 1
    
    mask2 = [1,2,4,128]
    for plane_number in range(len(mask2)):
        image_mask = image_component & mask2[plane_number]
        bin_component = image_mask >> plane_number
        
        ax = fig.add_subplot(3,5,idx)
        ax.imshow(bin_component, cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title('Plano de bit '+str(len(str(bin(mask2[plane_number]))[2:])-1))
        idx = idx + 1
        
plt.tight_layout()
plt.show()

####################### mostrar imagem com texto codificado ###############################
plt.imshow(image_encoded.copy() , cmap=plt.cm.gray)
plt.title('Imagem com mensagem oculta')
plt.axis('off')

fig, big_axes = plt.subplots( figsize=(15.0, 15.0) , nrows=3, ncols=1, sharey=True) 

for row, big_ax in enumerate(big_axes, start=1):
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    big_ax._frameon = False

idx = 1
for component in range(1,4):
    image_component = get_component(image_encoded.copy(),component-1)
    
    ax = fig.add_subplot(3,5,idx)
    ax.imshow(image_component, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_title('Canal '+ str(component)+' da imagem codificada')
    idx = idx + 1
    
    mask2 = [1,2,4,128]
    for plane_number in range(len(mask2)):
        image_mask = image_component & mask2[plane_number]
        bin_component = image_mask >> plane_number
        
        ax = fig.add_subplot(3,5,idx)
        ax.imshow(bin_component, cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title('Plano de bit '+str(len(str(bin(mask2[plane_number]))[2:])-1))
        idx = idx + 1
        
plt.tight_layout()
plt.show()


####################### guardar imagem com texto codificado ###############################
misc.imsave(output_img_file,image_encoded)


