import pickle
from q_table import QTable

def load_dict(filepath):
    """Lädt das Dictionary (die Superklasse) aus der gespeicherten Datei"""
    with open(filepath, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict



# Beispiel: Verwendung der Funktion
filepath = "example_qtable.pkl"  # Ersetze dies durch den tatsächlichen Dateipfad
loaded_dict = load_dict(filepath)
print(type(loaded_dict))
print(loaded_dict.__dict__)
print(isinstance(loaded_dict, dict))
print(isinstance(loaded_dict, QTable))
print(loaded_dict[ ((2, 0, 0, 0, 0, -2, 0, 4, 0, 0, 0, -2, -100, 0, 0, 0), 4)])

