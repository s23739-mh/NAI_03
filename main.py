"""
Rekomendator Filmów

Problem:
Stworzenie rekomendatora filmów na podstawie ocen użytkowników.

Autor:
[Mikolaj Pretki]
[Mikolaj Holdakowski]

Instrukcja Użycia:
1. Wczytaj dane z pliku JSON zawierające oceny filmów przez różnych użytkowników.
2. Przygotuj dane do modelu, przekształcając je do formy zrozumiałej przez algorytm.
3. Stwórz model rekomendacyjny (Random Forest Regressor) na podstawie ocen użytkowników.
4. Dla wybranego użytkownika, uzyskaj listę rekomendowanych i niepolecanych filmów.
5. Wyświetl wyniki rekomendacji.

Uwagi:
- Model trenowany jest na wszystkich dostępnych ocenach, a następnie stosowany do rekomendacji.
- Błąd średniokwadratowy jest używany do oceny jakości modelu (im niższy, tym lepiej).
- Filmy nieocenione są rekomendowane na podstawie przewidywanych ocen.
- Usunięto możliwość rekomendowania obejrzanych filmów oraz powtarzających się rekomendacji.
- Dodano 5 filmów nie rekomendowanych (ocena 1-3), eliminując powtórzenia.

Użyte importy:
sklearn (train_test_split, RandomForestRegressor, mean_squared_error, LabelEncoder)
numpy
json
"""

"""
Zmienne wejściowe:
- 'plik': nazwa pliku JSON zawierającego oceny filmów przez różnych użytkowników.
- 'uzytkownik': Imię i nazwisko dla którego wyświetlane będą rekomendacje
Dane wyjściowe:

Zmienne wyjściowe:
- 'rekomendacje': Lista filmów rekomendowanych
- 'nierekomendacje': Lista filmów nierekomendowanych
"""

import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np


plik = 'movie_data.json'
uzytkownik = "Mikolaj Pretki"

"""Wczytaj dane z pliku JSON"""
with open(plik, encoding='utf-8') as file:
    dane = json.load(file)

"""Przygotuj dane do uczenia modelu"""
oceny = []
filmy = []
for uzytkownicy, oceny_uzytkownika in dane.items():
    for film, ocena in oceny_uzytkownika.items():
        oceny.append(ocena)
        filmy.append(film)

"""Przekształcenie danych do trenowania modelu"""
le = LabelEncoder()
X = le.fit_transform(filmy).reshape(-1, 1)
y = np.array(oceny)

"""Podzielenie danych na zbiory treningowe i testowe"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Utworzenie modelu przy użyciu 'Random Forest Regressor'"""
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

"""Dokonanie predykcji na zbiorze testowym"""
y_pred = model.predict(X_test)

"""Ocena jakości modelu używając błędu średniokwadratowego"""
mse = mean_squared_error(y_test, y_pred)
print(f"Błąd średniokwadratowy: {mse}")

"""Przykład użycia modelu do rekomendacji dla danego użytkownika"""
filmy_uzytkownika = list(dane[uzytkownik].keys())
oceny_uzytkownika = list(dane[uzytkownik].values())
filmy_do_rekomendacji = [film for film in filmy if film not in filmy_uzytkownika]
oceny_do_rekomendacji = model.predict(le.transform(filmy_do_rekomendacji).reshape(-1, 1))

"""Wyświetlenie 5 rekomendacji, eliminując powtarzające się filmy"""
rekomendacje = set()
for i in np.argsort(oceny_do_rekomendacji)[::-1]:
    if filmy_do_rekomendacji[i] not in rekomendacje:
        rekomendacje.add(filmy_do_rekomendacji[i])
        if len(rekomendacje) == 5:
            break

print(f"Rekomendowane dla {uzytkownik}: {list(rekomendacje)}")

"""Wyświetlenie 5 filmów nierekomendowanych, eliminując powtarzające się filmy"""
nierekomendacje = set()
for i in np.argsort(oceny_do_rekomendacji):
    if filmy_do_rekomendacji[i] not in nierekomendacje:
        nierekomendacje.add(filmy_do_rekomendacji[i])
        if len(nierekomendacje) == 5:
            break
print(f"Nie rekomendowane dla {uzytkownik}: {list(nierekomendacje)}")