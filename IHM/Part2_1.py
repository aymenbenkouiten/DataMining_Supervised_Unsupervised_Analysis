from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import math
from collections import Counter
import time

dataset = pd.read_csv("datasets/Dataset2.csv")
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.fillna(dataset.median(), inplace=True)
dataset.head()

X = dataset.drop('Fertility', axis=1)
y = dataset['Fertility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
train_size = X_train.shape[0]
test_size = X_test.shape[0]
train_size, test_size

def distance_manhattan(instance1, instance2):
    return sum(abs(a - b) for a, b in zip(instance1, instance2))

def distance_euclidienne(instance1, instance2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(instance1, instance2)))

def distance_minkowski(instance1, instance2, p):
    return sum(abs(a - b) ** p for a, b in zip(instance1, instance2)) ** (1 / p)

def distance_cosine(instance1, instance2):
    dot_product = sum(a * b for a, b in zip(instance1, instance2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in instance1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in instance2))
    return 1 - (dot_product / (magnitude1 * magnitude2))

def distance_hamming(instance1, instance2):
    return sum(a != b for a, b in zip(instance1, instance2))

def trier_selon_distance(dataset, instance, distance_function):
    distances = [(data, distance_function(instance, data[:-1])) for data in dataset]
    distances.sort(key=lambda x: x[1])
    return distances

def classe_dominante(knn_instances):
    classes = [instance[-1] for instance, _ in knn_instances] #stock les classes des instances dans une list
    count = Counter(classes)
    return count.most_common(1)[0][0]

def predict_knn(training_set, instance, k, distance_function):
    distances = trier_selon_distance(training_set, instance, distance_function)
    knn = distances[:k]
    return classe_dominante(knn)

def KNN(training_set, test_x, k, distance_function):
    predictions = []
    for instance in test_x:
        prediction = predict_knn(training_set, instance, k, distance_function)
        predictions.append(prediction)
    return predictions

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def partition(rows, column, value):
    true_rows, false_rows = [], []
    for row in rows:
        if row[column] >= value:
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            true_rows, false_rows = partition(rows, col, val)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            p = float(len(true_rows)) / (len(true_rows) + len(false_rows))
            gain = current_uncertainty - p * gini(true_rows) - (1 - p) * gini(false_rows)

            if gain >= best_gain:
                best_gain, best_question = gain, (col, val)

    return best_gain, best_question

def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return class_counts(rows)

    true_rows, false_rows = partition(rows, question[0], question[1])
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return (question, true_branch, false_branch)

def classify(row, node):
    if not isinstance(node, tuple):
        return node
    question, true_branch, false_branch = node

    if row[question[0]] >= question[1]:
        return classify(row, true_branch)
    else:
        return classify(row, false_branch)

def DecisionTrees(training_set, test_set):
    tree = build_tree(training_set)
    predictions = []
    for instance in test_set:
        prediction = list(classify(instance, tree).keys())[0]
        predictions.append(prediction)
    return predictions

def bootstrap_sample(data):
    bootstrap = []
    for _ in range(len(data)):
        index = random.randint(0, len(data) - 1)
        bootstrap.append(data[index])
    return bootstrap

def random_forest_train(data, n_trees):
    trees = []
    for _ in range(n_trees):
        sample = bootstrap_sample(data)
        tree = build_tree(sample)
        trees.append(tree)
    return trees

def random_forest_predict(trees, row):
    predictions = [classify(row, tree) for tree in trees]
    final_prediction = max(predictions, key=predictions.count)
    return final_prediction

def RandomForest(training_set, test_set, n_trees):
    forest = random_forest_train(training_set, n_trees)
    predictions = []
    for instance in test_set:
        prediction = list(random_forest_predict(forest, instance).keys())[0]
        predictions.append(prediction)
    return predictions
    
def matrice_confusion(y_test, predictions):
    classes = y_test.drop_duplicates()

    matrice = [[0 for _ in classes] for _ in classes]
    class_to_index = {cls: i for i, cls in enumerate(classes)}

    for r, p in zip(y_test, predictions):
        actual_index = class_to_index[r]
        predicted_index = class_to_index[p]
        matrice[actual_index][predicted_index] += 1
    
    return matrice

def accuracy(confusion_matrix):
    correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total_predictions = sum(sum(row) for row in confusion_matrix)
    accuracy = correct_predictions / total_predictions
    return accuracy

def specificite(confusion_matrix, classe):
    total = sum(sum(row) for row in confusion_matrix)
    TN_FP = total - sum(confusion_matrix[classe])
    TN = TN_FP - sum(confusion_matrix[i][classe] for i in range(len(confusion_matrix)))
    FP = sum(confusion_matrix[i][classe] for i in range(len(confusion_matrix))) - confusion_matrix[classe][classe]
    return TN / (TN + FP) if (TN + FP) > 0 else 0

def precision(confusion_matrix, classe):
    TP = confusion_matrix[classe][classe]
    FP = sum(confusion_matrix[i][classe] for i in range(len(confusion_matrix))) - TP
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def rappel(confusion_matrix, classe):
    TP = confusion_matrix[classe][classe]
    FN = sum(confusion_matrix[classe]) - TP
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f_score(confusion_matrix, classe):
    prec = precision(confusion_matrix, classe)
    rapp = rappel(confusion_matrix, classe)
    return 2 * (prec * rapp) / (prec + rapp) if (prec + rapp) > 0 else 0

def evaluation(modele, training_set, test_set, y_test, **kwargs):
    start_time = time.time()
    predictions = modele(training_set, test_set, **kwargs)
    end_time = time.time()
    temps_execution = end_time - start_time


    eval = {}
    cm = matrice_confusion(y_test, predictions)
    cmTxt = ""
    for i in cm:
        for j in i:
            cmTxt += str(f"{j:03d}")+" "
        cmTxt += "\n"

    eval["Matrice de Confusion: "] = cmTxt

    exactitude_globale = accuracy(cm)
    specificite_globale, precision_globale, rappel_globale, f_score_globale = 0, 0, 0, 0
    for classe in range(len(cm)):
        spec = specificite(cm, classe)
        prec = precision(cm, classe)
        rapp = rappel(cm, classe)
        fscr = f_score(cm, classe)
        
        specificite_globale += spec
        precision_globale += prec
        rappel_globale += rapp
        f_score_globale += fscr
        eval["Classe "+str(classe)+" Specificite:"] = format(spec, '.2f')
        eval["Classe "+str(classe)+" Precision:"] = format(prec, '.2f')
        eval["Classe "+str(classe)+" Rappel:"] = format(rapp, '.2f')
        eval["Classe "+str(classe)+" F-Score:"] = format(fscr, '.2f')
    eval["Exactitude Globale:"] = format(exactitude_globale, '.2f')
    eval["Specificite Globale:"] = format(specificite_globale/len(cm), '.2f')
    eval["Precision Globale:"] = format(precision_globale/len(cm), '.2f')
    eval["Rappel Global:"] = format(rappel_globale/len(cm), '.2f')
    eval["F-Score Global:"] = format(f_score_globale/len(cm), '.2f')
    eval["Temps d'Execution:"] = str(format(temps_execution, '.2f'))+ "secondes"
    return eval
