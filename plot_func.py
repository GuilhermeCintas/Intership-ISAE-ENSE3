import matplotlib.pyplot as plt
import numpy as np


# ------------- Analyze 1 - Unprocessed Values -------------
"""lista = list()
cont = 0
palavra = 'recompensa'
with open("./slurm.1422484.txt") as dados:
  for line in dados:
    if palavra in line:
      a = float(line[11:].strip().replace("\x00",""))
      lista.append(a)
dados.close()
plt.plot(np.arange(0, np.size(lista), 1), lista)
plt.title("Reward")
plt.xlabel('episodes')
plt.ylabel('reward mean')
plt.grid()
plt.show()"""
# -------------------------------------------------------



import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

lista = list()
lista_media = list()
lista_aux = list()
cont = 0
palavra =  'recompensa'
#palavra =  'angulo'
#palavra =  'vel_ang'
#palavra =  'Distancia ponto-otimo'
#palavra =  'Distancia ponto-ponto'
#palavra =  'Distancia ponto-pfora'


with open("./slurm.1424725.txt") as dados:
  for line in dados:
    if palavra in line:
      if palavra == 'recompensa':
        a = float(line[11:].strip().replace("\x00",""))
      elif palavra == 'angulo':
        a = abs(float(line[7:].strip().replace("\x00","")))
      elif palavra == 'vel_ang':
        a = abs(float(line[8:].strip().replace("\x00","")))
      elif palavra == 'Distancia ponto-otimo' or 'Distancia ponto-ponto' or 'Distancia ponto-pfora':
        a = float(line[22:].strip().replace("\x00",""))
      else:
        print("Invalid information!!!")
      
      lista.append(a)
      lista_aux.append(a)
      cont += 1
      if np.size(lista_aux) == 1:
        cont = 0
        seila = np.mean(lista_aux)
        lista_media.append(seila)
        lista_aux.clear()
dados.close()
media_movel = moving_average(lista_media, 100)
plt.plot(np.arange(0, np.size(media_movel), 1), media_movel, label='Graph 1', color='tab:blue')

plt.xlabel('Episodes')

#plt.ylabel('Final reward value')
plt.ylabel('Angle')
#plt.ylabel('Rotation speed')
#plt.ylabel('Distance')
#plt.ylabel('Close distance reset')
#plt.ylabel('Long distance reset')
plt.legend()
plt.grid()




lista = list()
lista_media = list()
lista_aux = list()
cont = 0

with open("./slurm.1424732.txt") as dados:
  for line in dados:
    if palavra in line:
      if palavra == 'recompensa':
        a = float(line[11:].strip().replace("\x00",""))
      elif palavra == 'angulo':
        a = abs(float(line[7:].strip().replace("\x00","")))
      elif palavra == 'vel_ang':
        a = abs(float(line[8:].strip().replace("\x00","")))
      elif palavra == 'Distancia ponto-otimo' or 'Distancia ponto-ponto' or 'Distancia ponto-pfora':
        a = float(line[22:].strip().replace("\x00",""))
      else:
        print("Invalid information!!!")

      lista.append(a)
      lista_aux.append(a)
      cont += 1
      if np.size(lista_aux) == 1:
        cont = 0
        seila = np.mean(lista_aux)
        lista_media.append(seila)
        lista_aux.clear()
dados.close()
media_movel = moving_average(lista_media, 100)
plt.plot(np.arange(0, np.size(media_movel), 1), media_movel, label='Graph 2', color='tab:orange')


lista = list()
lista_media = list()
lista_aux = list()
cont = 0

with open("./slurm.1424868.txt") as dados:
  for line in dados:
    if palavra in line:
      if palavra == 'recompensa':
        a = float(line[11:].strip().replace("\x00",""))
      elif palavra == 'angulo':
        a = abs(float(line[7:].strip().replace("\x00","")))
      elif palavra == 'vel_ang':
        a = abs(float(line[8:].strip().replace("\x00","")))
      elif palavra == 'Distancia ponto-otimo' or 'Distancia ponto-ponto' or 'Distancia ponto-pfora':
        a = float(line[22:].strip().replace("\x00",""))
      else:
        print("Invalid information!!!")

      lista.append(a)
      lista_aux.append(a)
      cont += 1
      if np.size(lista_aux) == 1:
        cont = 0
        seila = np.mean(lista_aux)
        lista_media.append(seila)
        lista_aux.clear()
dados.close()
media_movel = moving_average(lista_media, 100)
plt.plot(np.arange(0, np.size(media_movel), 1), media_movel, label='Graph 3', color='tab:green')



lista = list()
lista_media = list()
lista_aux = list()
cont = 0

with open("./slurm.1424918.txt") as dados:
  for line in dados:
    if palavra in line:
      if palavra == 'recompensa':
        a = float(line[11:].strip().replace("\x00",""))
      elif palavra == 'angulo':
        a = abs(float(line[7:].strip().replace("\x00","")))
      elif palavra == 'vel_ang':
        a = abs(float(line[8:].strip().replace("\x00","")))
      elif palavra == 'Distancia ponto-otimo' or 'Distancia ponto-ponto' or 'Distancia ponto-pfora':
        a = float(line[22:].strip().replace("\x00",""))
      else:
        print("Invalid information!!!")

      lista.append(a)
      lista_aux.append(a)
      cont += 1
      if np.size(lista_aux) == 1:
        cont = 0
        seila = np.mean(lista_aux)
        lista_media.append(seila)
        lista_aux.clear()
dados.close()
media_movel = moving_average(lista_media, 100)
plt.plot(np.arange(0, np.size(media_movel), 1), media_movel, label='Graph 4', color='tab:red')

plt.xlabel('Episodes')

#plt.ylabel('Final reward value')
#plt.ylabel('Angle')
#plt.ylabel('Rotation speed')
#plt.ylabel('Distance')
#plt.ylabel('Close distance reset')
#plt.ylabel('Long distance reset')
plt.legend()
#plt.grid()

plt.show()