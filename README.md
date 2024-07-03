# World  of Warcraft - Battlegrounds
## Opis projektu
To jest pierwszy mój pierwszy projekt na nowej drodze data science. Jako iż to jest pierwszy chciałem, żeby był w tematyce, która uwielbiam. Projekt ten koncentruje się na analizie wyników bitew w grze World of Warcraft. Zawiera szczegółowe analizy wyników graczy w aktywnościach PVP (gracz vs gracz) oraz modele uczenia maszynowego (klasyfikacje oraz regresje), które pomogą zrozumieć zachodzące zależności w dostępnych danych.
## Cel projektu 
Celem projektu jest szczegółowa analiza wyników bitwy między frakcjami w grze. Analizy pokazują jak dane klasy postaci radziły sobie w aktywności PVP. Może to pomóc nowym graczom w wyborze klasy nowej postaci. Oprócz tego projekt zawiera modele klasyfikacji, które przewidują, do której frakcji nalezał gracz oraz jakie cechy najbardziej wypływały na tą decyzje. Podobny model jest zastosowany co do klasy z jakiej korzystał gracz. Jest rownież model regresji, która przwiduje zmiane cechy, która jest drugą z najważniejszych w aktywności PVP.
## Pliki i foldery
W repozytorium mamy notatniki jupytera oraz modele maszynowy w wersji python. Oto co zawierają:
- class_classification - zawiera modele klasyfikacji rozwiązujące problem przewidywania klasy jaką grał gracz oraz dane, które zostały przygotowane do uczenia tych modeli
- fraction_classification - zawiera modele klasyfikacji rozwiązujące problem przewidywania frakcji do której przystąpił gracz oraz dane, które zostały przygotowane do uczenia tych modeli
- hk_regression = zawiera modele regresji przewidujące jak zmienia się najważniejsza cecha dla graczy trybu PVP oraz dane, które zostały przygotowane do uczenia tych modeli
- analysis.ipynb - jest to notatnik, który zawiera analizę dostępnych danych oczyszczonych
- Preprocesing to ML.ipynb - notatnik, który pokazuje jak zostały przygotowane oczyszczone dane do uczenia maszynowego
- clean_data.csv - są to dane oczyszczone, gotowe do analizy.
- data_to_ml.csv - są to dane, które zostały przygotowane do modeli uczenia maszynowego. Kopia tego pliku jest w każdym z folderów.
- ml.ipynb -  jest to notanik, który zawiera kod modeli oraz wykresy takie jak (krzywa uczenia, macierz pomyłek)
- preprocesing data.ipnyb - jest to notatnik, który zawiera kroki w oczyszczeniu danych.
- wowbgs2.csv - surowe dane.
- presentation.pptx - prezentacja analiz danych i wyniokow modeli uczenia maszynowego. 
## Biblioteki
- NumPy version: 1.26.4
- Pandas version: 2.2.1
- Seaborn version: 0.13.2
- Matplotlib version: 3.8.3
- Scikit-learn version: 1.4.2
- Keras version: 3.3.3
- TensorFlow version: 2.16.1
