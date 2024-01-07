import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score
from prettytable import PrettyTable
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="DataMining Project", layout="wide")
st.title("DataMining Project")
def calculer_quartiles(attribut):
    # Trier l'attribut
    attribut_trie = sorted(attribut)

    # Calculer (Q0) qui est le min
    Q0 = attribut_trie[0]

    # Calculer le premier quartile (Q1)
    n = len(attribut_trie)
    q1_idx = n // 4
    Q1 = (attribut_trie[q1_idx - 1] + attribut_trie[q1_idx]) / 2

    # Calculer le deuxième quartile (Q2, la médiane)
    if n % 2 == 1:
        Q2 = attribut_trie[n // 2]
    else:
        mid_idx = n // 2
        Q2 = (attribut_trie[mid_idx - 1] + attribut_trie[mid_idx]) / 2

    # Calculer le troisième quartile (Q3)
    q3_idx = (3 * n) // 4
    Q3 = (attribut_trie[q3_idx - 1] + attribut_trie[q3_idx]) / 2

    # Calculer le quatrième quartile (Q4)
    q4_idx = n - q1_idx
    Q4 = (attribut_trie[q4_idx - 1] + attribut_trie[q4_idx]) / 2


    return Q0, Q1, Q2, Q3, Q4
def replace_outliers_mean(dfMoy, column_name):
    # Extraction de la colonne spécifiée
    column_data = dfMoy[column_name]
    
    # Calcul de la moyenne et de l'écart type de la colonne
    mean_value = column_data.mean()
    
    # Calcul des quartiles pour identifier les valeurs aberrantes
    q1 = column_data.quantile(0.25)
    q3 = column_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Remplacement des valeurs aberrantes par la moyenne
    dfMoy[column_name] = np.where((dfMoy[column_name] < lower_bound) | (dfMoy[column_name] > upper_bound), mean_value, dfMoy[column_name])
    return dfMoy 
def replace_outliers_median(dfMed, column_name):
    # Extraction de la colonne spécifiée
    column_data = dfMed[column_name]
    
    # Calcul de la moyenne et de l'écart type de la colonne
    
    # Définition des seuils pour les valeurs aberrantes
    q0, q1, median_value, q3, q4 = calculer_quartiles(column_data)
    iqr = q3 - q1
    mn = q1 - (1.5 * iqr)
    mx = q3 + (1.5 * iqr)
    
    # Remplacement des valeurs aberrantes par la median
    dfMed[column_name] = np.where((dfMed[column_name] < mn) | (dfMed[column_name] > mx), median_value, dfMed[column_name])
    return dfMed
def normalizationMinMax(data, min_old, max_old, min_new, max_new):
    normalized_data = [(x - min_old) / (max_old - min_old) * (max_new - min_new) + min_new for x in data]
    return normalized_data
def NormalizationZscore(data):
    moyenne = np.mean(data)
    ecart_type = np.std(data)

    # Normalise les données en utilisant la formule du score 
    donnees_normalisees = (data - moyenne) / ecart_type
    return donnees_normalisees
def visualize_clusters(instances, KMeans_Labels, centroides):
    # Instancier l'objet PCA
    pca = PCA(n_components=2)

    # Appliquer l'ACP sur les données normalisées
    pca_result = pca.fit_transform(instances)

    # Visualiser les résultats du K-means avec les deux premières composantes principales
    # plt.scatter(pca_result[:, 0], pca_result[:, 1], c=KMeans_Labels, cmap='viridis', edgecolors='k')
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=KMeans_Labels, cmap='coolwarm', edgecolors='k')
    plt.scatter(np.array(centroides)[:, 0], np.array(centroides)[:, 1], c='yellow', marker='X', s=200, label='Centroides')
    plt.title('K-means Clustering (PCA)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.show()

with st.sidebar:
    st.header("Navigation")
    selected_section = st.radio("Select Section", ["Analyse et prétraitement ds1", "Analyse et prétraitement ds2", "L’algorithme Apriori", "Analyse supervisée", "Analyse non supervisée"])
if selected_section == "Analyse et prétraitement ds1":
    st.header("Analyse et prétraitement ds1")
    
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv("datasets/Dataset1_1.csv")
    normalized = False
    col1, col2 = st.columns(2)
    if col1.button("Convertir les valeurs non numériques en NaN"):
        st.session_state.df['P'] = pd.to_numeric(st.session_state.df['P'], errors='coerce')
        
    if col1.button("Supprime les lignes a des valeurs manquantes"):
        st.session_state.df = st.session_state.df.dropna(subset=['P'])
        st.session_state.df = st.session_state.df.dropna(subset=['OC'])
        st.session_state.df = st.session_state.df.dropna(subset=['Cu'])
        normalized = True
    if col1.button("eliminer_redondance_lines"):
        st.session_state.df = st.session_state.df.drop_duplicates()
    if col1.button("eliminer_redondance_colonnes"):
        st.session_state.df = st.session_state.df.drop('OC', axis=1)
        st.session_state.df = st.session_state.df.drop('OM', axis=1)
    if col1.button("Boxplot"):
        col1.image("images\\ds1\\boxplot_before.png")
    col11, col12 = col1.columns(2)
    if col11.button("replace_outliers_mean"):
        for c in st.session_state.df.columns:
            st.session_state.df = replace_outliers_mean(st.session_state.df, c)
        col1.image("images\\ds1\\boxplot_after_moy.png")
    if col12.button("replace_outliers_median"):
        for c in st.session_state.df.columns:
            st.session_state.df = replace_outliers_median(st.session_state.df, c)
        col1.image("images\\ds1\\boxplot_after_med.png")
    if col11.button("normalizationMinMax"):
        min_old = st.session_state.df[st.session_state.df.columns[:-1]].min()
        max_old = st.session_state.df[st.session_state.df.columns[:-1]].max()  
        for colonne in st.session_state.df.columns[:-1]:
            st.session_state.df[colonne] = normalizationMinMax(st.session_state.df[colonne], min_old[colonne], max_old[colonne], 0, 1)
    if col12.button("NormalizationZscore"):
        for colonne in st.session_state.df.columns[:-1]:
            st.session_state.df[colonne] = NormalizationZscore(st.session_state.df[colonne])

    if col1.button("Visualisation des graphes"):
        col1.write("Histogrammes")
        col1.image("images\\ds1\\histogramme.png")
        col1.write("Matrice de Correlation")
        col1.image("images\\ds1\\matrice_correlation.png")
        
    col2.text("Dataset")
    col2.dataframe(st.session_state.df)
    col2.text("Description globale")
    col2.dataframe(st.session_state.df.describe())
    buffer = StringIO()
    st.session_state.df.info(buf=buffer)
    info_str = buffer.getvalue()
    if normalized:
        col2.text("Description de chaque attribut")
        table = []
        table.append(["Attribut", "Moyenne", "Médiane", "Mode", "Max", "Min", "q1", "q2", "q3"])
        col = st.session_state.df.columns
        for c in col:
            sorted_data = np.sort(st.session_state.df[c])
            q2 = round(np.median(sorted_data),2)
            q1 = round(np.percentile(sorted_data, 25),2)
            q3 = round(np.percentile(sorted_data, 75),2)
            moyenne = round(st.session_state.df[c].mean(),2)
            mediane = round(st.session_state.df[c].median(),2)
            mode = round(st.session_state.df[c].mode().values[0],2)
            max_val = st.session_state.df[c].max()
            min_val = st.session_state.df[c].min()
            table.append([c, moyenne, mediane, mode, max_val, min_val, q1, q2, q3])

        col2.dataframe(table)
elif selected_section == "Analyse et prétraitement ds2":
    st.header("Analyse et prétraitement ds2")
    def date_parser(x):
        date_formats = ['%d-%b', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                return pd.to_datetime(x, format=fmt)
            except ValueError:
                pass
        return pd.NaT
    if 'df2' not in st.session_state:
        st.session_state.df2 = pd.read_csv('datasets/Dataset1_2.csv', parse_dates=['Start date', 'end date'], date_parser=date_parser)
    normalized = False
    col1, col2 = st.columns(2)
    if col1.button("Convertir les valeurs non numériques en NaN"):
        for c in st.session_state.df2.columns:
            if(c== "Start date" or c== "end date"):
                continue
            st.session_state.df2[c] = pd.to_numeric(st.session_state.df2[c], errors='coerce')
        
    if col1.button("Supprime les lignes a des valeurs manquantes"):
        for c in st.session_state.df2.columns:
            st.session_state.df2 = st.session_state.df2.dropna(subset=[c])
        normalized = True
    if col1.button("eliminer_redondance_lines"):
        st.session_state.df2 = st.session_state.df2.drop_duplicates()
    if col1.button("Traittement des dates de mal format"):
        def map_year(time_period):
            if 21 <= time_period <= 35:
                return 2020
            elif 36 <= time_period <= 53:
                return 2021
            elif 54 <= time_period <= 155:
                return 2022
            else:
                return None
        st.session_state.df2['Start date'] = pd.to_datetime(st.session_state.df2['Start date'])
        st.session_state.df2['end date'] = pd.to_datetime(st.session_state.df2['end date'])
        st.session_state.df2['year'] = st.session_state.df2['time_period'].apply(map_year)
        mean_year = st.session_state.df2['year'].mean()
        st.session_state.df2['year'].fillna(mean_year, inplace=True)
        st.session_state.df2['end date'] = st.session_state.df2.apply(lambda row: row['end date'].replace(year=int(row['year'])), axis=1)
        st.session_state.df2['end date'] = st.session_state.df2['end date'].dt.strftime('%m/%d/%Y')
        st.session_state.df2['Start date'] = st.session_state.df2.apply(lambda row: row['Start date'].replace(year=int(row['year'])), axis=1)
        st.session_state.df2['Start date'] = st.session_state.df2['Start date'].dt.strftime('%m/%d/%Y')
    if col1.button("Boxplot"):
        col1.image("images\\ds2\\boxplot_avants.png")
    
    col11, col12 = col1.columns(2)
    if col11.button("replace_outliers_mean"):
        for c in st.session_state.df2.columns:
            if(c== "Start date" or c== "end date"):
                continue
            st.session_state.df2 = replace_outliers_mean(st.session_state.df2, c)
        col1.image("images\\ds2\\boxplot_apres_moy.png")
    
    if col12.button("replace_outliers_median"):
        for c in st.session_state.df2.columns:
            if(c== "Start date" or c== "end date"):
                continue
            st.session_state.df2 = replace_outliers_median(st.session_state.df2, c)
        col1.image("images\\ds2\\boxplot_apres_med.png")

    if col1.button("Visualisation des graphes"):
        graphs = ["Nombre total de cas confirmés par zone",
                  "Nombre total de tests positifs par zone",
                  "Répartition des cas confirmés par zone",
                  "Répartition des cas confirmés par zone..",
                  "évolution hebdomadaire des tests COVID-19, tests positifs et cas confirmés",
                  "évolution mensuelle des tests COVID-19, tests positifs et cas confirmés",
                  "évolution abbuelle des tests COVID-19, tests positifs et cas confirmés",
                  "Répartitiondes cas COVID-19 positifs par zone et par année",
                  "Rapport entre la population et le nombre de tests effectués",
                  "Les 5 zones les plus fortement impactées par le coronavirus",
                  "évolution des cas confirmés, Tests Effectués et Tests Positifs"]
        for index, graph_name in enumerate(graphs):
            col1.write(graph_name)
            col1.image("images\\ds2\\"+str(index+1)+".png")
            
    col2.text("Dataset")
    col2.dataframe(st.session_state.df2)
    col2.text("Description globale")
    col2.dataframe(st.session_state.df2.describe())
    buffer = StringIO()
    st.session_state.df2.info(buf=buffer)
    info_str = buffer.getvalue()
    if normalized:
        col2.text("Description de chaque attribut")
        table = []
        table.append(["Attribut", "Moyenne", "Médiane", "Mode", "Max", "Min", "q1", "q2", "q3"])
        col = st.session_state.df2.columns
        for c in col:
            sorted_data = np.sort(st.session_state.df2[c])
            q2 = round(np.median(sorted_data),2)
            q1 = round(np.percentile(sorted_data, 25),2)
            q3 = round(np.percentile(sorted_data, 75),2)
            moyenne = round(st.session_state.df2[c].mean(),2)
            mediane = round(st.session_state.df2[c].median(),2)
            mode = round(st.session_state.df2[c].mode().values[0],2)
            max_val = st.session_state.df2[c].max()
            min_val = st.session_state.df2[c].min()
            table.append([c, moyenne, mediane, mode, max_val, min_val, q1, q2, q3])

        col2.dataframe(table)   
elif selected_section == "L’algorithme Apriori":
    import Part1_3
    st.header("L’algorithme Apriori")
    df = pd.read_csv("datasets/Dataset2.csv")

    discretisation = st.radio("Méthod de discrétisation: ", ["Equal frequency", "Equal width"])
    bins = st.number_input("K", min_value=2, max_value=10, step=1)
    Min_Supp = st.number_input("Min_Supp", min_value=0, max_value=100, step=1)/100
    Min_Conf = st.number_input("Min_Conf", min_value=0, max_value=100, step=1)/100
    frequent_itemsets = []
    if st.button("Apply"):
        if discretisation == "Equal frequency":
            for attribute in ["Temperature", "Humidity", "Rainfall"]:
                df[attribute] = pd.to_numeric(df[attribute].str.replace(',', '.'), errors='coerce')
                df[attribute+'_equal_freq'] = Part1_3.equal_freq(df[attribute], bins, Part1_3.generate_labels(attribute, bins))
            df = df[[f'Temperature_equal_freq',f'Humidity_equal_freq', f'Rainfall_equal_freq', f'Soil', f'Crop', f'Fertilizer']]
        else:
            for attribute in ["Temperature", "Humidity", "Rainfall"]:
                df[attribute] = pd.to_numeric(df[attribute].str.replace(',', '.'), errors='coerce')
                df[attribute+'_equal_width'] = Part1_3.equal_width(df[attribute], bins, Part1_3.generate_labels(attribute, bins))
            df = df[[f'Temperature_equal_width',f'Humidity_equal_width', f'Rainfall_equal_width', f'Soil', f'Crop', f'Fertilizer']]

        frequent_itemsets = Part1_3.find_frequent_itemsets(df.values.tolist(), min_support=Min_Supp)
    if st.button("Extraction des fortes règles"):
        association_rules = Part1_3.generate_rules(frequent_itemsets,Min_Conf)
        association_rules = sorted(association_rules, key=lambda x: x[3], reverse=True)
        rules_with_correlation = Part1_3.correlation(association_rules)
elif selected_section == "Analyse supervisée":
    import Part2_1
    st.header("Analyse supervisée")
    st.write("Total: "+str(Part2_1.X.shape[0])+" Training set: "+str(Part2_1.train_size)+" Test set:"+str(Part2_1.test_size))

    algorithm = st.radio("Choose an algorithm", ["KNN", "Decision Trees", "Random Forest"])

    if algorithm == "KNN":
        k_parameter = st.number_input("K parameter", min_value=1, max_value=100, step=1)
        distance_function = st.selectbox("Distance Function", ["Euclidean", "Manhattan", "Hamming", "Cosine", "Minkowski"])
    elif algorithm == "Random Forest":
        number_of_trees = st.number_input("Number of Trees", min_value=10, max_value=1000, step=10)

    if st.button("Apply to The whole test set"):
        st.write("Éxecution de l'algorithme "+algorithm+"..")
        training_set = [list(x) + [y] for x, y in zip(Part2_1.X_train.values, Part2_1.y_train.values)]
        test_set = Part2_1.X_test.values
        if algorithm == "KNN":
            if distance_function=="Euclidean":
                distance_function = Part2_1.distance_euclidienne
            elif distance_function=="Manhattan":
                distance_function = Part2_1.distance_manhattan
            elif distance_function=="Hamming":
                distance_function = Part2_1.distance_hamming
            elif distance_function=="Cosine":
                distance_function = Part2_1.distance_cosine
            elif distance_function=="Minkowski":
                distance_function = Part2_1.distance_minkowski
            eval = Part2_1.evaluation(Part2_1.KNN, training_set, test_set, Part2_1.y_test, k=k_parameter, distance_function=distance_function)
        elif algorithm == "Decision Trees":
            eval = Part2_1.evaluation(Part2_1.DecisionTrees, training_set, test_set, Part2_1.y_test)
        elif algorithm == "Random Forest":
            eval = Part2_1.evaluation(Part2_1.RandomForest, training_set, test_set, Part2_1.y_test, n_trees=number_of_trees)
        for parm in eval:
            st.write(parm)
            st.text(eval[parm])

    st.subheader("Parameters")
    N = st.number_input("N", format="%f")
    P = st.number_input("P", format="%f")
    K = st.number_input("K", format="%f")
    pH = st.number_input("pH", format="%f")
    EC = st.number_input("EC", format="%f")
    OC = st.number_input("OC", format="%f")
    S = st.number_input("S", format="%f")
    Zn = st.number_input("Zn", format="%f")
    Fe = st.number_input("Fe", format="%f")
    Cu = st.number_input("Cu", format="%f")
    Mn = st.number_input("Mn", format="%f")
    B = st.number_input("B", format="%f")
    OM = st.number_input("OM", format="%f")

    if st.button("Apply"):
        st.write("Éxecution de l'algorithme "+algorithm+"..")
        training_set = [list(x) + [y] for x, y in zip(Part2_1.X_train.values, Part2_1.y_train.values)]
        test = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B, OM]
        if algorithm == "KNN":
            if distance_function=="Euclidean":
                distance_function = Part2_1.distance_euclidienne
            elif distance_function=="Manhattan":
                distance_function = Part2_1.distance_manhattan
            elif distance_function=="Hamming":
                distance_function = Part2_1.distance_hamming
            elif distance_function=="Cosine":
                distance_function = Part2_1.distance_cosine
            elif distance_function=="Minkowski":
                distance_function = Part2_1.distance_minkowski
            prediction = Part2_1.predict_knn(training_set, test, k=k_parameter, distance_function=distance_function)
        elif algorithm == "Decision Trees":
            tree = Part2_1.build_tree(training_set)
            prediction = list(Part2_1.classify(test, tree).keys())[0]
        elif algorithm == "Random Forest":
            forest = Part2_1.random_forest_train(training_set, number_of_trees)
            prediction = list(Part2_1.random_forest_predict(forest, test).keys())[0]
        st.write(prediction)
elif selected_section == "Analyse non supervisée":
    import Part2_2
    st.header("Analyse non supervisée")
    df = pd.read_csv('datasets/Dataset2.csv')
    true_labels = df['Fertility']
    df = df.drop('Fertility', axis=1)
    df = df.drop('OC', axis=1)
    df = df.drop('OM', axis=1)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    instances = df_scaled
    algorithm = st.radio("Algorithme: ", ["Kmeans", "Dbscan"])
    col1, col2 = st.columns(2)
    if algorithm == "Kmeans":
        if col1.button("Expérimentation des paramétres de k-means"):
            graphs = ["Méthode du coude pour déterminer k",
                      "2D Representation after PCA",
                      "Heatmap of Metric Values by Iteration",
                      "Heatmap of Metric Values by Convergence"]
            for index, graph_name in enumerate(graphs):
                col2.write(graph_name)
                col2.image("images\\knn\\"+str(index+1)+".png")
        k = st.number_input("K", min_value=0, max_value=10, step=1)
        max_iterations = st.number_input("max_iterations", min_value=100, max_value=10000, step=100)
        convergence_threshold = st.number_input("convergence_threshold", min_value=1e-2, max_value=1e-2, step=1e-2)
        
        if st.button("Apply"):
            instance_clusters, centroides = Part2_2.k_means(instances, k, max_iterations, convergence_threshold)
            df['Cluster'] = instance_clusters
            h_kmeans, c_kmeans, v_kmeans = homogeneity_completeness_v_measure(true_labels, df['Cluster'])
            st.write(f'Homogeneity {round(h_kmeans,4)}, Completeness {round(c_kmeans,4)}, V_measure {round(v_kmeans,4)}')
            silhouette_avg_sklearn = silhouette_score(df, df['Cluster'])
            st.write("Indice de silhouette moyen :", silhouette_avg_sklearn)
            visualize_clusters(instances, instance_clusters, centroides)
            df['Cluster'] = instance_clusters
            cluster_0_size = len(df[df['Cluster'] == 0])
            cluster_1_size = len(df[df['Cluster'] == 1])
            cluster_2_size = len(df[df['Cluster'] == 2])

            # Affichage de l'histogramme des tailles de clusters
            plt.figure(figsize=(5, 6))
            plt.bar(range(3), [cluster_0_size, cluster_1_size, cluster_2_size], color='lightcoral')
            plt.xticks(range(3), ['Cluster 0', 'Cluster 1', 'Cluster 2'])
            plt.title('Tailles des clusters après k-means')
            plt.xlabel('Cluster')
            plt.ylabel('Nombre d\'instances')
            st.pyplot()

    elif algorithm == "Dbscan":
        if col1.button("Expérimentation des paramétres de Dbscan"):
            graphs = ["DBSCAN Configuration Effect on silhouette score",
                      "DBSCAN Configuration Effect on silhouette score",
                      "Impact of EPS and Min Samples on Silhouette score"]
            for index, graph_name in enumerate(graphs):
                col2.write(graph_name)
                col2.image("images\\dbscan\\"+str(index+1)+".png")
        eps_value = st.number_input("eps_values", min_value=0, max_value=100, step=1)/100
        min_samples_value = st.number_input("min_samples_values", min_value=1, max_value=10, step=1)
        if st.button("Apply"):
            dbscan = Part2_2.DBSCAN(eps_value, min_samples_value)
            
            dbscan.fit(df)
            unique_labels = np.unique(dbscan.labels)
            if len(unique_labels) > 1:
                num_clusters = len(np.unique(dbscan.labels)) 
                silhouette_avg = silhouette_score(df, dbscan.labels)
                result = [eps_value, min_samples_value, silhouette_avg, num_clusters]
            table = PrettyTable()
            table.field_names = ['eps', 'min_samples', 'silhouette_score', 'num_clusters']
            table.add_row([result[0], result[1], result[2], result[3]])
            st.write(table)
            pca = PCA(n_components=2)

            pca_result = pca.fit_transform(df)

            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            df['DBSCAN_Labels'] =dbscan.labels
            plt.scatter(df['PCA1'], df['PCA2'], c=df['DBSCAN_Labels'], cmap='coolwarm', edgecolors='k')
            plt.title('DBSCAN Clustering (PCA)')
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            st.pyplot()