import laspy
import pandas as pd
import numpy as np
import os

DIR_PATH = os.path.dirname(__file__)
URL_CAPITOLE = DIR_PATH + "/toulouse_capitole.laz"
OUTPUT_CSV = DIR_PATH + "/toulouse_capitole.csv"

def laz_to_df(input_path: str, max_rows: int = 10000):
    """
    Convertit un fichier LAZ en fichier CSV contenant toutes les dimensions des points.

    Arguments:
    - input_path (str): Chemin du fichier LAZ d'entrée.
    - output_path (str): Chemin du fichier CSV de sortie.
    - max_rows (int): Nombre maximum de lignes à exporter dans le fichier CSV.
    """
    with laspy.open(input_path) as fh:
        print(f"Lecture du fichier LAZ : {input_path}")
        print(f"Points disponibles : {fh.header.point_count}")

        # Lire un nombre limité de points
        if max_rows == 0:
            data = fh.read()
        else:
            data = fh.read_points(max_rows)

        # Récupérer toutes les dimensions disponibles
        dimensions = fh.header.point_format.dimensions
        print(f"Dimensions trouvées : {[dim.name for dim in dimensions]}")

        # Construire un dictionnaire des données
        data_dict = {}
        for dim in dimensions:
            field = getattr(data, dim.name)
            # Convertir en liste si possible, sinon gérer comme chaîne
            if hasattr(field, "tolist"):
                data_dict[dim.name] = field.tolist()
            else:
                data_dict[dim.name] = [str(f) for f in field]

        # Convertir en DataFrame
        return pd.DataFrame(data_dict)

def sort_points_by_proximity(df: pd.DataFrame, x: float, y: float, z: float) -> pd.DataFrame:
    """
    Trie les points d'un DataFrame par ordre de proximité avec un point donné (x, y, z).

    Arguments:
    - df (pd.DataFrame): DataFrame contenant les colonnes X, Y, Z.
    - x (float): Coordonnée X du point de référence.
    - y (float): Coordonnée Y du point de référence.
    - z (float): Coordonnée Z du point de référence.

    Retourne:
    - pd.DataFrame: DataFrame trié par distance croissante au point (x, y, z).
    """
    # Calculer la distance euclidienne pour chaque point
    df['distance'] = np.sqrt((df['X'] - x) ** 2 + (df['Y'] - y) ** 2 + (df['Z'] - z) ** 2)
    # Trier par distance
    sorted_df = df.sort_values(by='distance').reset_index(drop=True)
    # Supprimer la colonne de distance pour ne pas polluer le DataFrame
    sorted_df = sorted_df.drop(columns=['distance'])
    return sorted_df

if __name__ == ("__main__"):
    df = laz_to_df(URL_CAPITOLE, max_rows=300000)

    # Sauvegarder en CSV
    # df.to_csv(OUTPUT_CSV, index=False)
    # print(f"Fichier CSV sauvegardé à : {OUTPUT_CSV}")

    # Exemple d'utilisation de la fonction de tri
    x_ref, y_ref, z_ref = 1000, 2000, 3000  # Coordonnées de référence
    sorted_df = sort_points_by_proximity(df, x_ref, y_ref, z_ref)
    print(sorted_df.head())