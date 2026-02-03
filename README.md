# 🎯 PROJET TABPFN SABR - README COMPLET ET CLAIR

## 📌 BUT DU PROJET (En Simple)

**Objectif Principal :** Améliorer TabPFN pour qu'il prédise mieux les volatilités SABR ET leurs dérivées (Greeks).

**Problème Identifié par Peter :**
> "TabPFN is quite good for the values, it struggles with the derivatives"

**Solution :**
1. Calculer les dérivées des volatilités SABR
2. Entraîner des modèles qui prédisent SIMULTANÉMENT volatilités + dérivées
3. Tester différentes activations pour trouver la meilleure
4. Comparer avec TabPFN baseline

---

## 🎯 WORKFLOW GLOBAL DU PROJET

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW COMPLET                            │
└─────────────────────────────────────────────────────────────────────┘

ÉTAPE 1 : BASELINE TABPFN (ce qui existe déjà)
┌──────────────────────────────────────────────────┐
│ Statap2_corrected.py                             │
│   → Génère données SABR (volatilités seulement) │
│   → Fichier : sabr_data_recovery.csv           │
│                                                  │
│ test_tabpfn.py                                  │
│   → Teste TabPFN sur ces données               │
│   → Résultat : MAE = 5e-5                      │
└──────────────────────────────────────────────────┘
                    ↓
                    
ÉTAPE 2 : AJOUT DES DÉRIVÉES (priorité Peter)
┌──────────────────────────────────────────────────┐
│ compute_derivatives.py                          │
│   → Génère NOUVELLES données avec dérivées     │
│   → Fichier : sabr_with_derivatives.csv        │
│   → Contient : volatilités + 6 dérivées        │
│                                                  │
│   ⚠️ REMPLACE sabr_data_recovery.csv           │
└──────────────────────────────────────────────────┘
                    ↓
                    
ÉTAPE 3 : ENTRAÎNER MODÈLES CUSTOM
┌──────────────────────────────────────────────────┐
│ ray_architecture_search.py                      │
│   → Lit : sabr_with_derivatives.csv            │
│   → Entraîne 30-50 modèles différents          │
│   → Utilise loss_with_derivatives.py           │
│   → Teste TOUTES activations (Mish, GELU...)   │
│   → Trouve la meilleure config                 │
│   → Fichier : best_config.json                 │
└──────────────────────────────────────────────────┘
                    ↓
                    
ÉTAPE 4 : COMPARAISON FINALE
┌──────────────────────────────────────────────────┐
│ final_evaluation.py                             │
│   → Compare TabPFN vs modèles custom           │
│   → Utilise best_config.json                   │
│   → Génère rapport final                       │
└──────────────────────────────────────────────────┘
```

---

## 📂 FICHIERS DU PROJET - EXPLICATION CLAIRE

### Groupe 1️⃣ : BASELINE (Déjà fait - Phase 1)

#### `base_sabr.py` et `hagan_2002_lognormal_sabr.py`
**Rôle :** Bibliothèques SABR (ne pas modifier)
**Utilité :** Utilisées par les autres fichiers pour calculer volatilités SABR

#### `Statap2_corrected.py`
**Rôle :** Génère données SABR (volatilités uniquement)
**Ce qu'il fait :**
```python
Pour chaque combinaison (beta, rho, volvol, v_atm, F):
    Pour chaque strike K:
        Calcule volatilité SABR
        
Résultat : CSV avec 5000 lignes
Colonnes : [beta, rho, volvol, v_atm_n, alpha, F, K, log_moneyness, volatility]
```

**Fichier créé :** `sabr_data_recovery.csv`

**⚠️ IMPORTANT :** Ce fichier sera **REMPLACÉ** par `compute_derivatives.py` !

#### `test_tabpfn.py`
**Rôle :** Test baseline de TabPFN
**Ce qu'il fait :**
```python
Charge sabr_data_recovery.csv
Entraîne TabPFN (modèle pré-entraîné)
Prédit volatilités
Calcule MAE
```

**Résultat :** MAE ≈ 5e-5 (excellent !)

---

### Groupe 2️⃣ : DÉRIVÉES (Phase 2 - Priorité Peter)

#### `compute_derivatives.py` ⭐⭐⭐
**Rôle :** GÉNÈRE DE NOUVELLES DONNÉES COMPLÈTES (volatilités + dérivées)

**Ce qu'il fait :**
```python
Pour chaque combinaison (beta, rho, volvol, v_atm, F):
    Pour chaque strike K:
        1. Calcule volatilité SABR
        2. Calcule ∂V/∂beta (dérivée par rapport à beta)
        3. Calcule ∂V/∂rho
        4. Calcule ∂V/∂volvol
        5. Calcule ∂V/∂v_atm_n
        6. Calcule ∂V/∂F (forward)
        7. Calcule ∂V/∂K (strike)
        
Résultat : CSV avec 5000 lignes
Colonnes : [beta, rho, volvol, ..., volatility, dV_dbeta, dV_drho, ...]
```

**Fichier créé :** `sabr_with_derivatives.csv`

**🔑 RÉPONSE À VOTRE QUESTION :**
> "Est-ce que statap2_corrected est inclus dans compute_derivatives ?"

**OUI !** `compute_derivatives.py` fait TOUT ce que fait `Statap2_corrected.py` PLUS les dérivées.

**Est-ce qu'on peut enlever Statap2_corrected ?**

**OUI !** On peut simplifier en 2 scénarios :

**SCÉNARIO A : Simple (utilise seulement volatilités)**
```
Statap2_corrected.py → sabr_data_recovery.csv → test_tabpfn.py
```

**SCÉNARIO B : Complet (utilise volatilités + dérivées)** ⭐ RECOMMANDÉ
```
compute_derivatives.py → sabr_with_derivatives.csv → tout le reste
```

**RÉPONSE :** Gardez les deux pour comparaison, mais **vous pouvez skip Statap2** et commencer directement avec `compute_derivatives.py` !

---

#### `loss_with_derivatives.py`
**Rôle :** Définit comment calculer l'erreur pendant l'entraînement

**POURQUOI CE FICHIER ?**

TabPFN baseline utilise une loss simple :
```python
loss = |volatilité_prédite - volatilité_vraie|
```

Nous voulons une loss qui inclut les dérivées :
```python
loss = |volatilité_prédite - volatilité_vraie| + 
       |dérivée_prédite - dérivée_vraie|
```

**Ce qu'il contient :**
- `DerivativeLoss` : Loss standard avec dérivées
- `WeightedDerivativeLoss` : Certaines dérivées comptent plus
- `HuberDerivativeLoss` : Robuste aux outliers
- `AdaptiveDerivativeLoss` : Poids qui s'adaptent pendant training

**🔑 RÉPONSE À VOTRE QUESTION :**
> "J'ai du mal à comprendre l'intérêt de loss_with_derivatives"

**Réponse :**
- TabPFN peut prédire volatilités correctement MAIS prédire mal les dérivées
- En ajoutant les dérivées dans la loss, on **FORCE** le modèle à apprendre AUSSI les pentes/gradients
- Résultat : Le modèle comprend la **forme complète** de la surface de volatilité, pas juste les points

**Utilisé par :** `ray_architecture_search.py` et `final_evaluation.py`

---

### Groupe 3️⃣ : RECHERCHE D'ARCHITECTURE (Phase 3)

#### `ray_architecture_search.py` ⭐⭐⭐
**Rôle :** Trouve automatiquement la MEILLEURE configuration de modèle

**Ce qu'il fait :**
```python
Pour 30-50 configurations différentes:
    Créer un modèle avec :
        - Activation aléatoire (Mish, GELU, Swish, ou SELU)
        - Architecture aléatoire (Transformer ou MLP)
        - Hyperparamètres aléatoires (learning rate, layers, etc.)
    
    Charger sabr_with_derivatives.csv
    Entraîner le modèle avec loss_with_derivatives
    Calculer MAE sur test set
    
Garder la MEILLEURE configuration
Sauvegarder dans best_config.json
```

**Fichier créé :** `ray_results/best_config.json`

**Exemple de best_config.json :**
```json
{
  "activation": "mish",
  "model_type": "transformer",
  "d_model": 256,
  "num_layers": 4,
  "learning_rate": 0.001,
  "batch_size": 64
}
```

**🔑 RÉPONSE À VOTRE QUESTION :**
> "On ne réutilise pas ce qui est fait précédemment ?"

**Si !** Ce fichier :
1. **Lit** `sabr_with_derivatives.csv` (généré par compute_derivatives.py)
2. **Utilise** les classes de loss dans `loss_with_derivatives.py`
3. **Génère** `best_config.json` utilisé par `final_evaluation.py`

**Chaîne de dépendances :**
```
compute_derivatives.py → sabr_with_derivatives.csv
                              ↓
loss_with_derivatives.py ←───┤
                              ↓
ray_architecture_search.py → best_config.json
                              ↓
                     final_evaluation.py
```

---

### Groupe 4️⃣ : ÉVALUATION (Phase 4)

#### `final_evaluation.py`
**Rôle :** Compare TOUS les modèles et génère le rapport final

**Ce qu'il fait :**
```python
1. Charge sabr_with_derivatives.csv
2. Teste TabPFN baseline (pour comparaison)
3. Entraîne modèle Transformer avec Mish
4. Entraîne modèle Transformer avec GELU
5. Entraîne modèle Transformer avec Swish
6. Entraîne modèle Transformer avec SELU
7. Compare tous les résultats
8. Génère tableaux et graphiques
```

**Fichiers créés :**
- `final_evaluation_results.csv` : Tableau comparatif
- `final_evaluation_report.md` : Rapport pour Peter
- `final_evaluation_plots.png` : Graphiques

**🔑 RÉPONSE À VOTRE QUESTION :**
> "On ne réutilise pas best_config.json ?"

**Bonne remarque !** Dans ma version actuelle, `final_evaluation.py` teste plusieurs configs prédéfinies.

**VERSION AMÉLIORÉE :** Il devrait charger `best_config.json` et tester cette config en priorité.

---

### Groupe 5️⃣ : AMÉLIORATIONS BONUS

#### `advanced_improvements.py`
**Rôle :** Techniques avancées optionnelles
**Contenu :** Data augmentation, ensemble, curriculum learning, etc.
**Utilité :** Bonus si vous voulez aller plus loin

#### `master_execution_guide.py`
**Rôle :** Lance tout automatiquement
**Utilité :** Au lieu de lancer chaque fichier manuellement

---

## 🔄 DÉPENDANCES ENTRE FICHIERS

```
┌─────────────────────────────────────────────────────────────┐
│                    GRAPHE DE DÉPENDANCES                     │
└─────────────────────────────────────────────────────────────┘

base_sabr.py ─────┐
                  ├──→ compute_derivatives.py
hagan_2002_*.py ──┘           │
                              ↓
                   sabr_with_derivatives.csv
                              │
                              ├──→ ray_architecture_search.py
                              │           │
                              │           ↓
loss_with_derivatives.py ─────┤    best_config.json
                              │           │
                              │           ↓
                              └──→ final_evaluation.py
                                          │
                                          ↓
                              final_evaluation_results.csv
                              final_evaluation_report.md
```

---

## 🎯 WORKFLOW SIMPLIFIÉ RECOMMANDÉ

### Option 1 : Workflow Complet (Recommandé)

```bash
# Étape 1 : Générer données avec dérivées
python compute_derivatives.py
# → Crée sabr_with_derivatives.csv

# Étape 2 : Trouver meilleure config automatiquement
python ray_architecture_search.py --data sabr_with_derivatives.csv --samples 30
# → Crée best_config.json

# Étape 3 : Évaluation finale
python final_evaluation.py --data sabr_with_derivatives.csv
# → Crée rapport final
```

**Durée totale :** 2-4 heures

### Option 2 : Workflow Rapide (Sans Ray Tune)

```bash
# Étape 1 : Générer données avec dérivées
python compute_derivatives.py

# Étape 2 : Évaluation directe (skip Ray Tune)
python final_evaluation.py --data sabr_with_derivatives.csv
```

**Durée totale :** 30-45 minutes

### Option 3 : Baseline Simple (Pour Comparaison)

```bash
# Étape 1 : Générer données baseline
python Statap2_corrected.py

# Étape 2 : Tester TabPFN baseline
python test_tabpfn.py
```

**Durée totale :** 5 minutes

---

## ❓ RÉPONSES À VOS QUESTIONS

### Q1 : "Est-ce que statap2_corrected est inclus dans compute_derivatives ?"

**Réponse : OUI !**

`compute_derivatives.py` génère :
- Toutes les colonnes de `Statap2_corrected.py` (volatilités)
- PLUS 6 colonnes supplémentaires (dérivées)

**Tableau comparatif :**

| Fichier | Colonnes | Nombre |
|---------|----------|--------|
| `Statap2_corrected.py` | beta, rho, volvol, v_atm_n, alpha, F, K, log_moneyness, **volatility** | 9 |
| `compute_derivatives.py` | beta, rho, volvol, v_atm_n, alpha, F, K, log_moneyness, **volatility, dV_dbeta, dV_drho, dV_dvolvol, dV_dvatm, dV_dF, dV_dK** | 15 |

### Q2 : "Est-ce qu'on peut enlever Statap2_corrected ?"

**Réponse : OUI, on peut simplifier !**

**Scénario Recommandé :**
1. Gardez `Statap2_corrected.py` uniquement pour tester TabPFN baseline rapidement
2. Utilisez `compute_derivatives.py` pour TOUT le reste du projet

**Workflow simplifié :**
```bash
# Comparaison baseline (optionnel)
python Statap2_corrected.py
python test_tabpfn.py

# ↓↓↓ PROJET PRINCIPAL ↓↓↓
python compute_derivatives.py
python ray_architecture_search.py
python final_evaluation.py
```

### Q3 : "Modifier test_tabpfn pour utiliser derivatives en entrée ?"

**Réponse : Oui mais NON recommandé.**

**Pourquoi ?**

TabPFN est un modèle **pré-entraîné** qui :
- Attend un certain nombre de features en entrée
- Est optimisé pour prédire UNE sortie
- Ne peut PAS prédire plusieurs sorties (volatilité + 6 dérivées)

**Solution :**
- Gardez `test_tabpfn.py` comme baseline (prédit seulement volatilité)
- Les nouveaux modèles custom (dans `ray_architecture_search.py`) prédisent volatilité + dérivées

**Comparaison :**
```
TabPFN :    [features] → [volatilité]
Nos modèles: [features] → [volatilité, dV_dbeta, dV_drho, ...]
```

### Q4 : "Peter a parlé de créer de la data avec des graphes comme dans le papier ?"

**Réponse : OUI, mais je ne l'ai PAS encore implémenté (c'est optionnel/avancé).**

**Ce que Peter veut dire :**

Dans le paper TabPFN, ils génèrent des **datasets synthétiques** en utilisant des **graphes causaux**.

**Exemple de graphe causal financier :**
```
Interest Rate → Bond Price
      ↓
Option Price ← Volatility → Strike
      ↓
    Greeks
```

**Ce que ça donnerait pour SABR :**
```python
# Définir relations causales
beta → volatility
rho → volatility
volvol → volatility
F → volatility → dV/dF
K → volatility → dV/dK
```

**Pourquoi Peter suggère ça :**
- Générer beaucoup plus de données variées
- Capturer les vraies relations causales
- Améliorer la généralisation

**Status :** C'est une **amélioration avancée** (Phase 5 optionnelle).

**Voulez-vous que je l'implémente ?** Ce serait un fichier supplémentaire : `causal_data_generation.py`

---

## 🎯 PROJET RÉORGANISÉ - VERSION CLAIRE

Suite à vos remarques, voici la structure SIMPLIFIÉE :

### Fichiers ESSENTIELS (Minimum)

```
1. compute_derivatives.py       # Génère TOUTES les données
2. loss_with_derivatives.py     # Définit loss pour entraînement
3. ray_architecture_search.py   # Trouve meilleure config
4. final_evaluation.py          # Compare et génère rapport
```

### Fichiers OPTIONNELS

```
5. Statap2_corrected.py        # Baseline rapide (optionnel)
6. test_tabpfn.py              # Test TabPFN baseline (optionnel)
7. advanced_improvements.py     # Techniques bonus (optionnel)
8. master_execution_guide.py   # Automatisation (optionnel)
```

---

## 🚀 COMMANDES POUR DÉMARRER

### Workflow Minimum (2-3 heures)

```bash
# 1. Installer dépendances
pip install torch tabpfn "ray[tune]" optuna scikit-learn pandas numpy matplotlib

# 2. Générer données complètes
python compute_derivatives.py
# Résultat : sabr_with_derivatives.csv

# 3. Recherche automatique
python ray_architecture_search.py --data sabr_with_derivatives.csv --samples 30
# Résultat : best_config.json

# 4. Évaluation finale
python final_evaluation.py --data sabr_with_derivatives.csv
# Résultat : rapport final pour Peter
```

### Workflow Rapide (30 min - Sans Ray Tune)

```bash
# 1. Générer données
python compute_derivatives.py

# 2. Évaluation directe
python final_evaluation.py --data sabr_with_derivatives.csv
```

---

## 📊 CE QUE VOUS OBTENEZ À LA FIN

### Résultats Concrets

1. **Fichier CSV avec données :** `sabr_with_derivatives.csv`
   - 5000 lignes
   - 15 colonnes (features + volatilité + 6 dérivées)

2. **Meilleure configuration :** `best_config.json`
   - Quelle activation fonctionne le mieux (Mish, GELU, etc.)
   - Quels hyperparamètres sont optimaux

3. **Rapport final :** `final_evaluation_report.md`
   - Comparaison TabPFN vs modèles custom
   - Tableaux de résultats
   - Recommandations pour Peter

4. **Preuves visuelles :** `final_evaluation_plots.png`
   - Graphiques de performance
   - Comparaison MAE

---

## 📝 MODIFICATIONS À FAIRE POUR CLARIFIER

Je vais créer **3 nouveaux fichiers corrigés** :

1. **`WORKFLOW_COMPLET.md`** - Flowchart visuel clair
2. **`compute_derivatives_standalone.py`** - Version all-in-one qui remplace Statap2
3. **`final_evaluation_improved.py`** - Version qui utilise best_config.json

**Voulez-vous que je les crée maintenant ?**

---

## ✅ RÉSUMÉ FINAL

**Le projet en 3 phrases :**
1. On génère des données SABR avec volatilités + dérivées
2. On entraîne des modèles qui prédisent les deux simultanément
3. On trouve quelle activation (Mish/GELU/etc.) marche le mieux

**Fichiers à lancer dans l'ordre :**
1. `compute_derivatives.py`
2. `ray_architecture_search.py`
3. `final_evaluation.py`

**Tout le reste est optionnel ou supportif !**

---

**Est-ce plus clair maintenant ? Ai-je répondu à toutes vos questions ?** 🎯
