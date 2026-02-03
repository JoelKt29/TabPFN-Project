# =============================================================================
# TABPFN  PROJECT OVERVIEW
# =============================================================================


# Main Objective:
# Improve TabPFN so that it predicts SABR implied volatilities, their derivatives and other financial data more accurately.

#
#
#
#
# Methodology : 
#
#
#
#
# Step 1 — TABPFN vs SABR 
#
#   -> Generating SABR data (volatilities only)
#   -> Tests TabPFN on the generated dataset
#   -> Target result: MAE ≈ 5e-5
#


# Step 2 — ADD DERIVATIVES 
#
#   -> Generating a new dataset including derivatives and step 1 variables
#   -> Improve loss function using derivatives



# STEP 3 — TRAIN CUSTOM MODELS
#
# 
#   -> Trains 50 different models with the derivatives dataset
#   -> Tests ALL activation functions (Mish, GELU, Swish, SELU, etc.)
#   -> Finds the best architecture and hyperparameters
#

# Step 4 —  OTHER DATASET

#
#   -> Generating causal graph data
#   -> Feeding our selected models
#
#


# Step 5 —  COMPARISON AND RESULTS
#
#   -> Compares original TabPFN vs finetuned models
#
#
# 
#  FILES 
# 
#
# 
#  Phase 1
# 
# base_sabr.py and hagan_2002_lognormal_sabr.py : Files found on Github used to compute SABR volatilities
#
#
# step3_SABR_market_data.py: Generates SABR data (volatilities only) for each combination (beta, rho, volvol, v_atm, F), for each strike K: it will compute SABR volatility
#
#   CSV with ~5000 rows
#
# Columns:
#   [beta, rho, volvol, v_atm_n, alpha, F, K, log_moneyness, volatility]
#
# Created file:
#   sabr_data_recovery.csv
#
# test_tabpfn.py
# Role:
#   TabPFN baseline test
#
# What it does:
#
#   load sabr_data_recovery.csv
#   train TabPFN (pre-trained model)
#   predict volatilities
#   compute MAE
#
# Result:
#   MAE ≈ 5e-5 (excellent)
# -----------------------------------------------------------------------------



# =============================================================================
# GROUP 2 — DERIVATIVES (Phase 2 – Peter’s priority)
# =============================================================================

# -----------------------------------------------------------------------------
# compute_derivatives.py
# Role:
#   GENERATES NEW COMPLETE DATA (volatilities + derivatives)
#
# What it does:
#
#   for each combination (beta, rho, volvol, v_atm, F):
#       for each strike K:
#           1. compute SABR volatility
#           2. compute dV/dbeta
#           3. compute dV/drho
#           4. compute dV/dvolvol
#           5. compute dV/dv_atm_n
#           6. compute dV/dF (forward)
#           7. compute dV/dK (strike)
#
# Result:
#   CSV with ~5000 rows
#
# Columns:
#   [beta, rho, volvol, ..., volatility, dV_dbeta, dV_drho, ...]
#
# Created file:
#   sabr_with_derivatives.csv
#
#
# Frequently asked question:
#   "Is Statap2_corrected included inside compute_derivatives?"
#
# Answer:
#   YES.
#   compute_derivatives.py does EVERYTHING Statap2_corrected.py does,
#   PLUS all derivatives.
#
#
# Can we remove Statap2_corrected?
#
# YES. Two possible scenarios:
#
# Scenario A — Simple (volatility only)
#   Statap2_corrected.py → sabr_data_recovery.csv → test_tabpfn.py
#
# Scenario B — Full (volatility + derivatives)  ← RECOMMENDED
#   compute_derivatives.py → sabr_with_derivatives.csv → everything else
#
# In practice:
#   You can skip Statap2 and directly start with compute_derivatives.py
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# loss_with_derivatives.py
# Role:
#   Defines how the training error is computed
#
# Why this file?
#
# TabPFN baseline uses a simple loss:
#   loss = |predicted_vol - true_vol|
#
# We want a loss that also includes derivatives:
#   loss = |predicted_vol - true_vol|
#        + |predicted_derivative - true_derivative|
#
# What it contains:
#   - DerivativeLoss           : standard derivative-aware loss
#   - WeightedDerivativeLoss   : some derivatives weighted more heavily
#   - HuberDerivativeLoss      : robust to outliers
#   - AdaptiveDerivativeLoss   : weights adapt during training
#
# Key idea:
#   TabPFN may predict values well but poorly estimate slopes.
#   Adding derivatives forces the model to learn the full geometry
#   of the volatility surface, not only pointwise values.
#
# Used by:
#   ray_architecture_search.py
#   final_evaluation.py
# -----------------------------------------------------------------------------



# =============================================================================
# GROUP 3 — ARCHITECTURE SEARCH (Phase 3)
# =============================================================================

# -----------------------------------------------------------------------------
# ray_architecture_search.py
# Role:
#   Automatically finds the BEST model configuration
#
# What it does:
#
#   for 30–50 different configurations:
#       create model with:
#           - random activation (Mish, GELU, Swish, SELU)
#           - random architecture (Transformer or MLP)
#           - random hyperparameters (lr, layers, etc.)
#
#       load sabr_with_derivatives.csv
#       train with loss_with_derivatives
#       compute MAE on validation set
#
#   keep the BEST configuration
#   save it to best_config.json
#
# Created file:
#   ray_results/best_config.json
#
# Example:
#   {
#       "activation": "mish",
#       "model_type": "transformer",
#       "d_model": 256,
#       "num_layers": 4,
#       "learning_rate": 0.001,
#       "batch_size": 64
#   }
#
# Dependencies:
#
#   compute_derivatives.py → sabr_with_derivatives.csv
#                                  ↓
#   loss_with_derivatives.py  ←────┤
#                                  ↓
#   ray_architecture_search.py → best_config.json
#                                  ↓
#                          final_evaluation.py
# -----------------------------------------------------------------------------



# =============================================================================
# GROUP 4 — EVALUATION (Phase 4)
# =============================================================================

# -----------------------------------------------------------------------------
# final_evaluation.py
# Role:
#   Compares ALL models and generates the final report
#
# What it does:
#
#   1. load sabr_with_derivatives.csv
#   2. evaluate TabPFN baseline
#   3. train Transformer + Mish
#   4. train Transformer + GELU
#   5. train Transformer + Swish
#   6. train Transformer + SELU
#   7. compare all results
#   8. generate tables and plots
#
# Created files:
#   final_evaluation_results.csv
#   final_evaluation_report.md
#   final_evaluation_plots.png
#
# Improvement suggestion:
#   Load best_config.json and evaluate that configuration first
# -----------------------------------------------------------------------------



# =============================================================================
# GROUP 5 — BONUS IMPROVEMENTS
# =============================================================================

# -----------------------------------------------------------------------------
# advanced_improvements.py
# Role:
#   Optional advanced techniques
#
# Content:
#   data augmentation, ensembling, curriculum learning, etc.
#
# Use:
#   Extra improvements if you want to go further
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# master_execution_guide.py
# Role:
#   Runs the entire pipeline automatically
#
# Use:
#   Instead of executing each script manually
# -----------------------------------------------------------------------------
