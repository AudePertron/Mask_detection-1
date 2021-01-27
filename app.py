from tensorflow.keras import models
import cv2
import numpy as np

#Chargement du model et demarrage de la Webcam
model = models.load_model('reco_masque')
video = cv2.VideoCapture(0)

# #Pour avoir les dimensions de la fenêtre
# frame_width  = video.get(cv2.CAP_PROP_FRAME_WIDTH )
# frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT )
# print('frame dimensions (HxW):',int(frame_height),"x",int(frame_width))

#Boucle de capture d'image
while True:

    #Lecture de la video capture
    ret, frame = video.read()

    #Traitement sur l'image capturé
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (124, 124))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    #Prédiction de la capture video
    prediction = model.predict_classes(img_array)[0][0]
    # prediction = int(model.predict(img_array)[0][0])
    print(prediction)

    # Choix et affichage du smiley en fonction de la prédiction
    reaction = None
    if prediction == 0:
        reaction = cv2.imread('./images/angry.png')
    else:
        reaction = cv2.imread('./images/happy.png')


    #Concaténation de la fenêtre de video capture et la fenêtre avec le smiley
    # cv2.imshow("Capturing", frame)
    # cv2.imshow("Reaction", reaction)
    window = np.concatenate((reaction, frame), axis=0)
    cv2.imshow("Detecteur de Masque", window)

    #Si on appui sur la touche "q" la webcam s'éteint
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

#Nettoyage
video.release()
cv2.destroyAllWindows()