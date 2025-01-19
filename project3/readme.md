Ziel:
Trainieren eines Reinforcement learning agents für ein 2D, 4x4-Gitter "Rock-Paper-Scissor"-Spiel mit zwei Löchern und einem Zielfeld.
Als Umgebung verwenden wir die vorgegebene gymnasium-Umgebung im Ordner "rps_game"
In der Klasse RPSEnv im modul "rps_game.env" wurde die Zeile set_opponent(self,opponent) hinzugefügt, um bei einer Umgebung den Gegner nachträglich ändern zu können (verwendet in stable_baseline_models_with_action_mask_selftraining.py: model.env.env_method("set_opponent",op)).

Im Modul utilities.training_and_evaluation werden die Funktionen evaluation und training definiert. Mit der evaluation-Funktion werden zwei Modelle gegeneinander ausgepielt und die Zahl an Gewinne und Niederlagen angegeben.
Mit der training-Funktionen werden die Modelle trainiert.

Abgegeben werden nur die fertigen zwei Modelle; bei Bedarf können die restlichen Modelle 

Probierte Algorithmen:
1. Q learning Table: 
    Programm in "q_table.py",
    Beispiel in "q_learning_example.py",

    Modell wird gegen den RandomOpponent vom modul "rps_game.opponent" trainiert. 
    Das trainierte Modell hat eine Gewinnrate von 99.6% gegen den vorgegebenen RandomOpponent. 

    Weiterführung in "q_table_continue_training_against_itself.py": Das Modell wird gegen sich selbst trainiert.
    Die Resultat ist, dass das trainierte Modell eine hohe Gewinnrate gegen den Gegner beitzt, gegen den es trainiert wurde, bei den RandomOpponent jedoch keine gute Gewinnrate mehr besitzt. 
    Dies deutet nach unserer Meinung darauf hin, dass das Modell, gegen welches trainiert wird, "auswendig gelernt" wird, anstatt dass die Spielmechanik "verstanden" wird. 
    -> Plot von wins in out_qtable

2. Algorithmen implementiert von Stable Baselines3:
    Die für dieses Spiel anwendbaren Algorithmen sind A2C, DQN und PPO (Discrete Action Space), jedoch ist bei diesen Algorithmen eine direkte Maskierung der Aktionen nicht möglich, die nicht gültig sind, z.B. eine Bewegung, die zum Verlassen des Spielfelds führen würde.
    
    Deswegen wird im Modul "utilities.RPSEnvWrapper" ein Wrapper für die Umgebung implementiert, welcher eine ungültige Aktion mit einem negativen Reward bestraft und durch eine zufällige Aktion ersetzt.
    Der Nachteil hiervon ist, dass gelernt werden muss, welche Aktionen ungültig sind.
    
3. MaskablePPO von Stable Baselines3:
    Eine Implementation, die den Nachteil des Wrappers nicht besitzt, ist die MaskablePPO, die es ermöglicht, ungültige Aktionen zu maskieren. Somit werden ungültige Aktionen ausgeschlossen und es muss nicht mehr gelernt werden, welche Aktionen gültig sind und welche nicht.
    
Durchführung mit MaskablePPO:
Die default Netzwerkstruktur hat zwei hidden Layers mit jeweils 64 Neuronen.
Beim Trainieren wird ein Modell anfänglich gegen den RandomOpponent über mehrere Epochen trainiert. Das durchtrainierte Modell wird als eine Generation bezeichnet. Bei folgenden Generationen wird der Gegner durch die vorherige Generation ersetzt.

Wir testen mit der default Netzwerkstruktur, wie die Lernparameter eingestellt werden müssen, um ein gutes Training durchzuführen. Zu den Lernparametern gehören die Lernrate und der Discount-Faktor "gamma".

Wir trainieren 80 Generationen für eine custom Netzwerkstruktur von a) drei hidden Layers mit jeweils 128 Neuronen, b) drei hidden Layers mit jeweils 512 Neuronen, und 23 Generationen für die Netzwerkstruktur von drei hidden Layers mit jeweils 1024 Neuronen.

Ein Test mit dem Modul utilities.training_and_evaluation für die verschiedenen Modelle legen nahe, dass die Modelle mit 512 Neuronen die besten Resultaten erzielen.
-> csv Tabelle
-> 512-Neuronen: epsilon reward plot

Wir wählen zwei Modelle aus (Generation 16 und 21), die beide dieselbe Netzwerkstruktur von 512 Neuronen besitzen und die besten Resultate haben.


test2vs2: parallelisiertes Ausrechnen. Anzahl an parallelisierte Jobs geben Zahl der Kämpfe an, die zur selben Zeit stattfinden. Dies verkürzt die Zeit für das Evaluieren
Ein Time-out von 100s wird festgestetzt, um potenziell unendlich dauernde Spiele zu unterbrechen




