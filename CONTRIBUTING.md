# 🛠 Protocole de Contribution Git - Version Officielle

Ce document définit les règles de gestion du code et de collaboration pour le projet. Le respect de ce protocole est obligatoire pour garantir la traçabilité du travail de chaque membre.

### 1. Gestion des Branches 

*Main Branch* : La branche main doit toujours contenir un code stable ve fonctionnel. Interdiction de "push" directement sur main sans test préalable.

*Feature Branches* : Chaque nouvelle tâche (ex: EDA, SQL, ML) doit être développée sur une branche séparée.

| Nommage | exemple |
| :--- | :--- |
| **feature/[nom-de-la-tache]** | `feature/sql-setup` |
| **feature/[nom-de-la-tache]** | `feature/eda-vitamin-a` |


### 2. Workflow Quotidien

Pour éviter les conflits de version, suivez cet ordre précis :

0) Synchronisation : git pull origin main (Commencer la journée avec le code à jour).
1) Développement : Effectuer les modifications locales.
2) Indexation : git add . (Préparer les fichiers).
3) Validation : git commit -m "Type: Description" (Enregistrer les changements).
4) Publication : git push origin [votre-branche].

### 3. Standard des Messages de Commit (Importent)


| Préfixe | Usage | Exemple |
| :--- | :--- | :--- |
| **Feat** | Ajout d'une nouvelle fonctionnalité | `Feat: ajout du script de prédiction` |
| **Fix** | Correction d'une erreur ou d'un bug | `Fix: correction du lien de la base SQL` |
| **Data** | Modification des datasets ou de la DB | `Data: nettoyage du fichier CIQUAL.csv` |
| **Doc** | Documentation, README, rapports | `Doc: rédaction de la problématique` |
| **Style** | Changement esthétique (CSS, UI, format) | `Style: mise à jour du design Streamlit` |
| **Refactor** | Optimisation du code (sans changer la fonction) | `Refactor: optimisation de la boucle de tri` |

### 4. Revue et Merge (Birleştirme)

Une fois une tâche terminée, une **Pull Request (PR)** doit être ouverte sur GitHub.

Le code sera vérifié par **au moins un autre membre** avant d'être fusionné dans la branche main.

### 5. Livrables et Traçabilité

Fréquence : Un commit sans push ne compte pas. Chaque membre doit pousser son travail à la fin de chaque séance.

Responsabilité : Les statistiques GitHub serviront de base pour évaluer l'implication individuelle dans le projet.