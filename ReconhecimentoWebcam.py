import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np  
import shutil     
import sys

pula_quadros =10
cap = cv2.VideoCapture(0)
cont_quadros =0
limiar = 0.3
detectorFaces = dlib.get_frontal_face_detector()
detectorPontosFaciais = dlib.shape_predictor('Recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('Recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load("Recursos/descritores_Alunos_Uniube.pickle", allow_pickle = True)
descitoresFaciais = np.load("Recursos/descritores_Alunos_Uniube.npy")
while cap.isOpened():
    ret, frame = cap.read()
    cont_quadros +=1
    if(cont_quadros % pula_quadros ==0):
        facesDetectadas = detectorFaces(frame, 1)
        for faces in facesDetectadas:
            e, t, d, b = (int(faces.left()), int(faces.top()), int(faces.right()), int(faces.bottom()))

            pontosFaciais =  detectorPontosFaciais(frame, faces)
            descritoFacial = reconhecimentoFacial.compute_face_descriptor(frame, pontosFaciais)
            listaDescritorFacial = [ df for df in descritoFacial ]
            npArrayDescitores = np.asarray(listaDescritorFacial, dtype=np.float64)
            npArrayDescitores = npArrayDescitores[np.newaxis, :]


            distancias = np.linalg.norm(np.array(npArrayDescitores) - np.array(descitoresFaciais), axis=1)
            minimo  = np.argmin(distancias)
            distancia_minima = distancias[minimo]
            pecentual = 100 - (distancias[minimo] * 100)

            if distancia_minima <= limiar:
                nome = os.path.split(indices[minimo])[1].split(".")[0]
            else:
                nome = ''

            cv2.rectangle(frame, (e, t), (d, b), (0, 0, 255), 2)
            texto = "{} {:.2f} %".format(nome, pecentual)
            cv2.putText(frame, texto, (d , t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0))
        cv2.imshow("Fotos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
sys.exit(0)



