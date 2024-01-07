import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import math
from tabulate import tabulate
import numpy as np
import random
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

def calcule_centroide(instances):
    if not instances:
        raise ValueError("Instances list is empty")
    num_dimensions = len(instances[0])
    somme = [0] * num_dimensions
    for instance in instances:
        for i in range(num_dimensions):
            somme[i] += instance[i]
    moyenne = [s / len(instances) for s in somme]
    return moyenne

def calcule_distance_euclidienne(A, B):
    distance = 0
    for i in range(len(A)):
        distance = distance + (A[i] - B[i])**2
    return round(math.sqrt(distance),2)    

def initialize_centroids_kmeans(instances, k):
    centroids = [random.choice(instances)]
    while len(centroids) < k:
        distances = np.array([min(np.linalg.norm(np.array(instance) - np.array(centroid)) ** 2 for centroid in centroids) for instance in instances])
        probabilities = distances / sum(distances)
        next_centroid = random.choices(instances, probabilities)[0]
        centroids.append(next_centroid)
    return centroids

def k_means(instances, k, max_iterations=100, convergence_threshold=1e-4):
    if k <= 0:
        raise ValueError("Invalid number of clusters or empty dataset")
    centroides = initialize_centroids_kmeans(instances, k)
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for i, instance in enumerate(instances):
            distances = [calcule_distance_euclidienne(instance, centroid) for centroid in centroides]
            closest_cluster_index = distances.index(min(distances))
            clusters[closest_cluster_index].append(i)
        new_centroides = [calcule_centroide([instances[i] for i in cluster]) for cluster in clusters]
        variation = np.sum((np.array(new_centroides) - np.array(centroides)) ** 2)
        if variation < convergence_threshold:
            break
        centroides = new_centroides
    instance_clusters = [-1] * len(instances)
    for cluster_index, cluster in enumerate(clusters):
        for instance_index in cluster:
            instance_clusters[instance_index] = cluster_index
    return instance_clusters, centroides

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps  # La distance maximale entre deux points pour les considérer comme voisins
        self.min_samples = min_samples  # Le nombre minimum de points dans un voisinage pour former un cluster
        self.labels = None  # Stocke les étiquettes de cluster pour chaque point

    def fit(self, df):
        self.labels = [0] * len(df)  # Initialisation des étiquettes à 0 pour chaque point
        cluster_id = 0  # Initialisation de l'identifiant de cluster
        self.core_samples = []  # Liste pour stocker les indices des points centraux (core points)

        # Parcours de chaque point dans le dataframe
        for i in range(len(df)):
            if self.labels[i] != 0:  # Si le point a déjà une étiquette de cluster
                continue

            neighbors = self.get_neighbors(df, i)  # Obtenir les voisins du point
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Marquer comme bruit (points isolés)
            else:
                cluster_id += 1  # Nouvel identifiant de cluster
                self.core_samples.append(i)  # Ajouter l'indice du point central (core point)
                self.expand_cluster(df, i, neighbors, cluster_id)  # Étendre le cluster

    def get_neighbors(self, df, index):
        neighbors = []
        for i in range(len(df)):
            if self.distance(df.iloc[index], df.iloc[i]) < self.eps:
                neighbors.append(i)
        return neighbors


    def expand_cluster(self, df, index, neighbors, cluster_id):
        self.labels[index] = cluster_id  # Attribue l'identifiant de cluster au point actuel

        i = 0
        while i < len(neighbors):  # Parcours des voisins actuels du point
            neighbor = neighbors[i]

            if self.labels[neighbor] == -1:  # Si le voisin est actuellement marqué comme bruit
                self.labels[neighbor] = cluster_id  # Réassigne au même cluster que le point actuel

            elif self.labels[neighbor] == 0:  # Si le voisin n'est pas attribué à un cluster
                self.labels[neighbor] = cluster_id  # Attribue le même identifiant de cluster au voisin
                new_neighbors = self.get_neighbors(df, neighbor)  # Récupère de nouveaux voisins pour ce voisin

                if len(new_neighbors) >= self.min_samples:  # Si le nombre de nouveaux voisins est suffisant pour former un cluster
                    neighbors = neighbors + new_neighbors  # Étend la liste des voisins pour inclure ces nouveaux voisins

            i += 1  # Passe au voisin suivant dans la liste des voisins actuels


    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)  # Calcul de la distance euclidienne entre deux points

    def calculate_intra_cluster_density(self, df):
        intra_cluster_density = 0
        total_points = len(df)

        for i in range(total_points):
            if self.labels[i] != -1:  # Ignorer les points marqués comme bruit
                neighbors = self.get_neighbors(df, i)
                intra_cluster_density += len(neighbors)  # Ajouter la taille du voisinage

        core_points_count = len(self.core_samples)
        if core_points_count > 0:
            intra_cluster_density /= core_points_count  # Calculer la densité moyenne des clusters

        return intra_cluster_density

    def calculate_inter_cluster_density(self, df):
        inter_cluster_density = 0
        total_points = len(df)

        for i in range(total_points):
            for j in range(i + 1, total_points):
                if self.labels[i] != self.labels[j]:  # Points dans des clusters différents
                    inter_cluster_density += 1 / self.distance(df.iloc[i], df.iloc[j])

        if total_points > 1:
            inter_cluster_density /= (total_points * (total_points - 1) / 2)  # Calculer la densité moyenne entre clusters

        return inter_cluster_density
