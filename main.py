import pandas as pd
import pyvista as pv

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from pyvista import GPUInfo
gpu_info = GPUInfo()
print(gpu_info)

def process_and_visualize_dataframe(df: pd.DataFrame, y_true, y_pred):
    """
    Visualise les données d'un DataFrame avec PyVista, permettant de switcher entre les couleurs basées sur les classifications correctes et prédites.

    Arguments:
    - df (pd.DataFrame): DataFrame contenant les colonnes X, Y, Z, et classification.
    - y_true (array-like): Tableau des classifications correctes.
    - y_pred (array-like): Tableau des classifications prédites.
    """
    # Extraire les colonnes nécessaires pour la visualisation
    points = df[['X', 'Y', 'Z']].values

    # Création d'un nuage de points avec PyVista
    cloud = pv.PolyData(points)

    # Définir les couleurs basées sur les classifications correctes
    color_map = {
        1: [255, 255, 255],  # Blanc
        2: [105, 105, 105],  # Gris foncé
        5: [0, 255, 0],      # Vert
        6: [255, 0, 0]       # Rouge
    }
    
    colors_true = [color_map.get(cls, [0, 0, 0]) for cls in y_true]  # Noir par défaut si non défini
    colors_pred = [color_map.get(cls, [0, 0, 0]) for cls in y_pred]

    # Ajouter les couleurs initiales (correctes)
    cloud['colors'] = colors_true

    # Affichage interactif avec PyVista
    plotter = pv.Plotter()

    label_actor = plotter.add_text("Result", position="upper_left", font_size=12, color="black")

    def switch_colors(state):
        """Change les couleurs entre les valeurs correctes et prédites."""
        nonlocal label_actor
        if state:  # Activer les couleurs prédites
            cloud.point_data['colors'] = colors_pred
            plotter.remove_actor(label_actor)
            label_actor = plotter.add_text("Pred", position="upper_left", font_size=12, color="black")
        else:  # Revenir aux couleurs correctes
            cloud.point_data['colors'] = colors_true
            plotter.remove_actor(label_actor)
            label_actor = plotter.add_text("Result", position="upper_left", font_size=12, color="black")

    # Ajouter un bouton pour switcher entre prédiction et résultat
    plotter.add_checkbox_button_widget(
        switch_colors,
        value=False,
        color_on="white",
        color_off="gray"
    )

    # Ajouter le nuage de points au plotter
    plotter.add_mesh(cloud, scalars="colors", rgb=True, point_size=5, render_points_as_spheres=True)
    plotter.show()

if __name__ == ("__main__"):
    # Charger le CSV dans un DataFrame
    CSV_PATH = "toulouse_capitole.csv"
    df = pd.read_csv(CSV_PATH)

    df = df.drop(['synthetic', 'key_point', 'withheld', 'overlap', 'edge_of_flight_line', 'scanner_channel', 'scan_direction_flag', 'user_data', 'point_source_id', 'gps_time'], axis=1)
    df = df[df['classification'].isin([6, 2, 5, 1])]

    # Diviser les données en X et y
    y = df['classification']
    X = df.drop('classification', axis=1)


    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Créer et entraîner le modèle
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Prédire les résultats
    y_pred = knn.predict(X_test)

    # Afficher les métriques
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Préparer les données pour la visualisation
    X_test['classification'] = y_test.values  # Ajouter les classifications correctes
    process_and_visualize_dataframe(X_test, y_test.values, y_pred)
