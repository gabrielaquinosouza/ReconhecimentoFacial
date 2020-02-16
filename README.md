# ReconhecimentoFacial
Reconhecimento de Faces

Passo a Passo
1º Passo - 
instale a biblioteca Dlib

2º Passo  - 
Baixe os arquivos:
shape_predictor_68_face_landmarks.dat
dlib_face_recognition_resnet_model_v1.dat

3º Passo
No arquivo de Treinamento Mude o nome da pasta ou caminho onde estão as imagens de treinamento na linha 17,
Sempre colocar o nome da imagem com o nome da pessoa dona da face da imagem, caso na imagem haver mais de uma pessoa a imagem sera movida para pasta de teste

4º Passo - 
No arquivo de Teste mude o nome da pasta onde contém as imagens com as faces para teste, linha 16

5º Passo - 
Execute o Arquivo ReconhecimentoWebcam.py para reconhecer faces via Webcam


*Obs: Controle a acurácia do reconhecimento pelo parâmetro limiar, quanto menor o limiar mais preciso será o algoritmo
