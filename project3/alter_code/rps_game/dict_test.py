class CustomDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_method(self):
        # Eine eigene Methode
        return {key: len(str(value)) for key, value in self.items()}

    def __setitem__(self, key, value):
        # Beispiel für überschriebenes Verhalten
        if not isinstance(key, str):
            raise TypeError("Keys müssen Strings sein.")
        super().__setitem__(key, value)

    def __setitem__(self, key, value):
        # Beispiel für überschriebenes Verhalten
        if not isinstance(key, str):
            raise TypeError("Keys müssen Strings sein.")
        super().__setitem__(key, value)


    def __getitem__(self, key):
        # Beispiel für benutzerdefinierte Logik beim Zugriff
        if key not in self:
            return f"{key} ist nicht im Dictionary!"
        return super().__getitem__(key)

custom_dict = CustomDict(a=1, b=2, c=3)
print(custom_dict)  # Ausgabe: {'a': 1, 'b': 2, 'c': 3}

custom_dict['d'] = 4
print(custom_dict['d'])  # Ausgabe: 4

# Nutzung der benutzerdefinierten Methode
print(custom_dict.custom_method())  # Ausgabe: {'a': 1, 'b': 1, 'c': 1, 'd': 1}

# Versuch, einen Key hinzuzufügen, der kein String ist
try:
    custom_dict[1] = "value"
except TypeError as e:
    print(e)  # Ausgabe: Keys müssen Strings sein.

# Zugriff auf einen nicht existierenden Key
print(custom_dict['e'])  # Ausgabe: e ist nicht im Dictionary!
