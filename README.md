# Mask_detection

## Contexte du projet

Avec la pandémie COVID-19, dans pas mal de pays le port de masques est devenu obligatoire pour se protéger contre ce virus mortel. Dans ce projet, vous alllez développer un modèle IA - Détecteur de masque sur les images en premier temps puis en temps réel avec Python.

## Description des données

Le dataset initial est `Dataset_masks`, composé de plusieurs sous-dossier, `train` et `test`, tout deux composés de deux autres sous-dossier contenant nos images, `with_mask` et `without_mask`. Avant de former et entrainer notre modèle, nous avons retraités ces images (conversion en noir et blanc, et redimensionnement (124x124 px) ) puis enregistré celles-ci dans le dossier `traitement` sous la même architecture que dans `Dataset_masks`. 


## Description de l'architecture du modèle

Les données passent d'abord par une normalisation puis passent trois fois de suite par une couche de Conv2D et une couche de maxpooling. Enfin, nous leur appliquons un Flatten pour les remettre à plat et deux couches Dense (ReLu puis sigmoid) pour récupérer le bon format de sortie (un booléen : 0 ou 1). Pour la compilation du modèle, étant donné que nous sommes dans un problème de classification binaire (0 ou 1), nous avons choisi une validation croisé binaire (`binary_crossentrop`) avec l'optimiseur `RMSprop`.
Nous obtenons une précision de 92%.


## Conclusion

Nous avons testé deux réseaux différents (bien qu'assez similaires) l'un utilisant des activations de type 'reLu', et l'autres sans activations. Les résultats sur la matrice de confusion allaient clairement en faveur du modèle 'reLu', c'est donc celui que nous avons choisi de conserver.
Nous avons quand même réalisé que notre modèle manquait d'entraînement (la base de données d'images était trop petite et pas assez diversifiée) et par conséquent les résultats sont mitigés. Le modèle ne détecte correctement que les visages convenablement illuminés (ni trop ni trop peu), bien cadrés et proches de la caméra. Nous avons également pu constater qu'en plaçant des visages à un endroit précis de l'écran, il pouvait détecter la présence d'un masque même s'il n'y en avait pas.
Notre modèle est probablement correct, mais ses résultats seraient optimisés avec une base d'apprentissage plus grande.


## L'application : 

La detection vidéo se fait depuis le fichier `app.py`. Si le modèle detecte le non port de masque, le smiley dans la fenêtre supérieur est mécontent, et inversement lorsque le masque est porté. 

