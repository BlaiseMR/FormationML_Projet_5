from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns

def get_words(lab, delimiter):
    words=[]
    
    if type(lab) is str :
        words = lab.split(delimiter)
    elif type(lab) is list :
        for word in lab :
            if type(word) is str :
                words += word.split(delimiter)
    
    return words

def most_common_words(labels, delimiter, top):
    words = []
    count = []
    for lab in labels:
        words += get_words(lab, delimiter)
    counter = Counter(words)
    words = []
    i = 0
    for word in counter.most_common(top):
        i = i + 1
#         print(i, word)
        words.append(word[0])
        count.append(word[1])
    return count, words

def check_common_words(labels, delimiter, top):
    words = []
    col1 = []
    col2 = []
    col3 = []
    
    for lab in labels:
            words += get_words(lab, delimiter)
    counter = Counter(words)
    for word in counter.most_common(top):
        col1.append(word[0])
        col2.append(word[1])
        col3.append(word[1] * 100 / len(labels))
        
    df2 = pd.DataFrame({'Words' : col1, 'Counts' : col2, 'Ratio' : col3})
    df2 = df2.sort_values(by = 'Counts', axis = 0, ascending = False)
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.gca()
    ax = df2.plot.bar(x='Words', y='Counts', rot=90)
    i=0
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(df2.iloc[i]['Ratio']), (p.get_x() * 1.005, p.get_height() * 1.005), rotation=45 )
        i+=1
    ax.set_title(labels.name, fontsize=14)
    plt.xlabel('Words')
    plt.ylabel('count')
    
    return df2

def check_words(labels, ref_words, delimiter):
    words=[]
    words = get_words(labels, delimiter)
    for word in ref_words:
        if word in words:
            return True
    return False

def select_words(labels, ref_words, delimiter):
    words=[]
    words = get_words(labels, delimiter)
    for word in ref_words:
        if word in words:
            return True
    return False

def eta_squared(y,x):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni':len(yi_classe),
                       'moyenne_classe' : yi_classe.mean()
                       })
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT

def ANOVA(df, X, Y):
    sous_echantillon = df[df[X].notnull()].sort_values(by=[Y])
    modalites = sous_echantillon[Y].unique()
    groupes = []
    
    for m in modalites:
        groupes.append(sous_echantillon[sous_echantillon[Y]==m][X])
        
    medianprops = {'color': 'black'}
    meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}
    
    eta = eta_squared(sous_echantillon[X],sous_echantillon[Y])
    print(r"$\eta^2 = {:.2f}$".format(eta))
    
    fig, ax = plt.subplots(figsize=(7,6))
    plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianprops,
               vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
    ax.set_title(label=r"{} : {} ($\eta^2 = {:.2f}$)".format(X,Y,eta), fontsize=14)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
#             plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
#             plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    
    fig = plt.figure(figsize=(7,6))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
#     plt.show(block=False)
    
def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()
    
def continges_table(df, X, X_vec, Y, Y_vec):

    fig = plt.figure(figsize=(7,6))
    new_df = df[df[X].isin(X_vec)]
    new_df = new_df[new_df[Y].isin(Y_vec)]

    print(len(new_df))
    
    cont = new_df[[X,Y]].pivot_table(index=X, columns=Y, aggfunc=len, margins=True, margins_name='Total')
    cont = cont.fillna(0)

    tx = cont.loc[:,['Total']]
    ty = cont.loc[['Total'],:]
    n = len(new_df)
    indep = tx.dot(ty)/n

    measure = (cont-indep)**2/indep
    xi_n = measure.sum().sum()
    table = measure/xi_n
    sns.heatmap(table.iloc[:-1,:-1], annot=cont.iloc[:-1,:-1])
    plt.show()
    
    return cont