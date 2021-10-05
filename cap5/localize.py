#!/usr/bin/env python

"""
Este programa implementa un freno de emergencia para evitar accidentes en Duckietown.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2
from pupil_apriltags import Detector

def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    action = actions.get(key, np.array([0.0, 0.0]))
    return action


if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='4tags')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            map_name = args.map_name,
            domain_rand = False,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    # se inicializa el detector
    at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

    # valor de f encontrado en el desafio anterior
    f = 862.5
    
    h = 0.042
    area = -1

    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(0)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        # Se define la acción dada la tecla presionada
        action = mov_duckiebot(key)


        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()

        # convertir imagen a escala de grises
        img_gris = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # detectar los tags
        tags = []
        tags = at_detector.detect(img_gris, estimate_tag_pose=False, camera_params=None, tag_size=None)
        for tag in tags:
            # obtener esquinas del tag
            (x1,y1,x2,y2) = tag.corners
            x1 = (int(x1[0]), int(x1[1]))
            y1 = (int(y1[0]), int(y1[1]))
            x2 = (int(x2[0]), int(x2[1]))
            y2 = (int(y2[0]), int(y2[1]))
            print(y1)
            print(y2)
            # calcular p usando las esquinas encontradas (la altura de la deteccion)
            p = y1[1]-y2[1]

            # calcular la distancia desde el robot hasta la deteccion
            dist = f*h/p
            print(dist)
            if dist <= 1:
                #si se esta cerca del tag, asignar su id a la variable area
                area = tag.tag_id
                pass

            #dibujar la deteccion en la imagen
            cv2.line(obs, x1, y1, (0, 255, 0), 2)
            cv2.line(obs, y1, x2, (0, 255, 0), 2)
            cv2.line(obs, x2, y2, (0, 255, 0), 2)
            cv2.line(obs, y2, x1, (0, 255, 0), 2)
            
        print(f"ubicacion actual: area {area}")
        cv2.imshow('patos', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.imshow('gris', img_gris)
    # Se cierra el environment y termina el programa
    env.close()
