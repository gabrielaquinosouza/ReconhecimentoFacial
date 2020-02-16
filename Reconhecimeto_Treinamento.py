import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np  
import shutil     


detectorFaces = dlib.get_frontal_face_detector()
detectorPontosFaciais = dlib.shape_predictor('Recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('Recursos/dlib_face_recognition_resnet_model_v1.dat')
indice = {}
idx = 0
descritoresFaciais = None

for arquivos in glob.glob(os.path.join('Treinamento' , '*jpg')):
    imagem = cv2.imread(arquivos)
    facesDetectadas = detectorFaces(imagem, 2)
    numFacesDetectadas = len(facesDetectadas)
    if numFacesDetectadas == 0:
       print("Nenhuma Face Detectada na imagem {}! A mesma Sera excluida".format(arquivos))
       os.remove(arquivos)
       print("excluido!!!!")
    elif numFacesDetectadas > 1:
       print("Mais de uma face detectada na imagem {}! A mesma sera movida para pasta Teste ".format(arquivos))
       shutil.move(arquivos, '/home/gabriel/PycharmProjects/ReconhecimentoFacial/Teste')
       print("Movido!!!!")
    for face in facesDetectadas:
        pontosFaciais = detectorPontosFaciais(imagem, face)
        descritoreFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [ df for df in descritoreFacial ]
        npArrayDescitores = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescitores = npArrayDescitores[np.newaxis, :]
        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescitores
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescitores), axis=0)
        
        indice[idx] = arquivos
        idx += 1 
print(descritoresFaciais.shape)
np.save("Recursos/descritores_Alunos_Uniube.npy", descritoresFaciais)
with open("Recursos/descritores_Alunos_Uniube.pickle", 'wb') as f:
    cPickle.dump(indice, f)   
     