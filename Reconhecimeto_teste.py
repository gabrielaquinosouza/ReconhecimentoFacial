import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np  
import shutil     

limiar = 0.529
detectorFaces = dlib.get_frontal_face_detector()
detectorPontosFaciais = dlib.shape_predictor('Recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('Recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load("Recursos/descritores_Alunos_Uniube.pickle", allow_pickle = True)
descitoresFaciais = np.load("Recursos/descritores_Alunos_Uniube.npy")
print(descitoresFaciais)
for arquivos in glob.glob(os.path.join('Teste' , '*jpg')):
    imagem = cv2.imread(arquivos)
    facesDetectadas = detectorFaces(imagem, 2)

    for faces in facesDetectadas:
        e, t, d, b = (int(faces.left()), int(faces.top()), int(faces.right()), int(faces.bottom()))

        pontosFaciais =  detectorPontosFaciais(imagem, faces)
        descritoFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [ df for df in descritoFacial ]
        npArrayDescitores = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescitores = npArrayDescitores[np.newaxis, :]


        distancias = np.linalg.norm(npArrayDescitores - descitoresFaciais, axis=1)
        minimo  = np.argmin(distancias)
        distancia_minima = distancias[minimo]

        if distancia_minima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = ''

        cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 2)
        texto = "{} {:.4f}".format(nome, distancia_minima)
        cv2.putText(imagem, texto, ((d - e) /2 , t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0))

    cv2.imshow("Fotos", imagem)
    cv2.waitKey(0)
cv2.destroyAllWindows()