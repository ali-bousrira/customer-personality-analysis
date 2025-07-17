Veille sur les méthodes de sélection du nombre optimal de clusters et la qualité d’un cluster

Introduction

En clustering, la sélection du nombre optimal de clusters est essentielle pour obtenir des résultats pertinents et faciles à interpréter. Différentes méthodes existent pour déterminer ce nombre et évaluer la qualité des clusters obtenus. Voici un état des lieux simplifié et clair sur ce sujet.

1. Méthodes de sélection du nombre optimal de clusters

a. Méthode du coude (Elbow Method)
Principe : Identifier visuellement un « coude » sur une courbe représentant la variation de la dispersion (distance intra-cluster) selon le nombre de clusters.
Avantage : Facile à comprendre et à appliquer.
Limite : Assez subjective et parfois difficile à interpréter.

<img width="290" height="174" alt="image" src="https://github.com/user-attachments/assets/959852cf-4a85-416a-8671-9a1d3132cf7a" />


b. Score de Silhouette
Principe : Calcule la qualité de séparation des clusters, en évaluant à la fois leur cohésion interne et leur séparation externe. Un score élevé indique un bon clustering.
Avantage : Intuitif, facile à interpréter.
Limite : Peut devenir lent à calculer sur de gros volumes de données.

c. Indice Davies-Bouldin
Principe : Évalue le rapport entre la dispersion interne des clusters et leur séparation externe. Plus l'indice est faible, meilleur est le clustering.
Avantage : Rapide à calculer, simple à comprendre.
Limite : Moins performant avec des clusters de tailles et formes variées.

d. Indice Calinski-Harabasz
Principe : Basé sur le rapport entre variance inter-clusters et variance intra-clusters. Une valeur élevée signifie un clustering efficace.
Avantage : Simple et efficace.
Limite : Performances réduites avec des clusters non-sphériques ou des données bruitées.

e. Méthodes statistiques (AIC/BIC)
Principe : Utilisent des modèles probabilistes (ex : mélanges gaussiens) pour sélectionner le nombre de clusters en cherchant un équilibre entre ajustement et complexité.
Avantage : Objectif et rigoureux statistiquement.
Limite : Complexité élevée, nécessite un cadre probabiliste clair.

2. Mesures de qualité d’un cluster
   
a. Mesures internes (sans référence externe)
Ces mesures évaluent la qualité intrinsèque du clustering :
Score de Silhouette (déjà décrit).
Indice Davies-Bouldin (déjà décrit).
Indice Dunn : Distance minimale entre clusters divisée par la plus grande dispersion interne. Valeur élevée = bon clustering.

b. Mesures externes (avec référence externe)
Utilisées quand la vraie répartition est connue (validation supervisée) :
Adjusted Rand Index (ARI) : Mesure de similarité entre les résultats du clustering et la vérité terrain.
Normalized Mutual Information (NMI) : Indique à quel point les résultats du clustering correspondent aux labels réels (entre 0 et 1).
V-measure : Combine homogénéité et complétude en une seule mesure claire et efficace.

3. Nouvelles approches

Gap Statistic : Compare la dispersion observée à celle attendue sous une hypothèse nulle (distribution aléatoire). Précis mais coûteux en temps de calcul.
Stabilité du clustering : Mesure la stabilité du clustering sous perturbations des données pour choisir le nombre optimal de clusters.

Deep clustering : Utilise des réseaux de neurones profonds pour améliorer le clustering sur des données complexes.

Conclusion
Il est recommandé d’utiliser plusieurs méthodes complémentaires pour choisir le nombre optimal de clusters et évaluer leur qualité, en tenant compte de la taille et de la complexité des données traitées.
