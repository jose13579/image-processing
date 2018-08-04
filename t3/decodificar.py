#coding: utf-8

# Para rodar o codigo correr, por:
#    python decodificar.py image_file output_text plane_number
# Exemplo:
#    python decodificar.py image1_saida.png texto_saida.txt 0

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

# Mostrar todos os planos de bits
# entrada: uma componente da imagem de entrada, numero da componente
def show_images_plane(image_component,nro_component):  
    mask = [1,2,4,128]
    # para cada valor do numero de planos de bits
    for plane_number in range(len(mask)):
	# realizar uma operaçao logical a cada componente da imagem com o plano de bits
        image_mask = image_component & mask[plane_number]

	# desplazar plano de bits posiçoes para obter o valor do bit na posiçao plano de bits
        bin_component = image_mask >> plane_number
    
	# apresentar o plano de bit para cada componente 
        plt.imshow(bin_component, cmap=plt.cm.gray)
        plt.xlabel('Canal '+str(nro_component)+' da Imagem no numero de plano de bits '+str(len(str(bin(mask[plane_number]))[2:])-1))
        plt.show()

# Obter a messagem oculta em codigo binario 
# entrada: palabvra
def get_cod_bin_word(word):
    merge_bin = ""
    for character in word:
        character_ascii = ord(character) # transformar caracter para codigo ASCII
        ascii_bin = bin(character_ascii) # transformar codigo ASCII para codigo binario
        merge_bin += ascii_bin[2:].zfill(8) # tirar os dois primeiros caracteres e cheiar até que o codigo binario tenha tamanho oito

    #retornar codigo binario da palavra
    return merge_bin


# Escrever a palavra no arquivo de texto
# entrada: nome do arquivo de saida
def write_word(output_text,word):
    #write it
    csvfile = open(output_text, 'w')
    csvfile.write(word)
        #writer.writerow((word))

# Metodo que decodifica a messagem no bit menos significativo
# entrada: imagem com messagem codificada, plano de bits e texto saida
# saida: palavra decodificada
def get_word_decoded_image(image_encoded, plane_number, mask):
    cod_bin_word = ""
    mensagem = "" 
    indx = 0
    flag = False
    
    # para cada pixel de cada componente da imagem
    for w in range(image_encoded.shape[0]):
        for h in range(image_encoded.shape[1]):   
            for c in range(3):
		# Para cada pixel da imagem com a mensagem oculta
                pixel_value = image_encoded[w,h,c]

		# realizar uma operaçao logica a cada pixel da componente da imagem com o plano de bits
                image_mask = pixel_value & mask[plane_number]

		# desplazar plano de bits posiçoes para obter o valor do bit na posiçao plano de bits
                bin_value = image_mask >> plane_number
                
		# concatenar cada bit na posiçao plano de bit para obter a mansagem em codigo binario
                cod_bin_word += str(bin_value)
                
		# aumentar o contador
                indx= indx + 1
                       
		# se o contador mod 8 == 0 transformar o codigo binario 8 bits a um caracter 
                if(indx%8 == 0): 
		    # se o caracter é igual a "]" parar o loop e mudar o flag
		    # caso contrario transformar o byte(8 bits) para codigo ASCII
		    word = chr(int(cod_bin_word, 2))
                    if(chr(int(cod_bin_word, 2)) == "]"):
                        flag = True
                        break
                    else:
                        mensagem += word
                        cod_bin_word = ""
                    
            if(flag==True): 
                break
                
        if(flag==True): 
            break
            
    # retornar a mensagem oculta
    return mensagem



img_file = sys.argv[1]
output_text = sys.argv[2]
plane_number = int(sys.argv[3])
mask = [1,2,4,8,16,32,64,128]

# Ler o arquivo da imagem
image = misc.imread(img_file)

# Presentaçao da imagem
plt.imshow(image, cmap='gray')
plt.show()

################################ Codificar palavra ###################################
# Encoded word in the image
word_decoded = get_word_decoded_image(image.copy(), plane_number, mask)


####################### Mostrar palavra ###############################
print "Mensagem: ",word_decoded






