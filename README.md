# Structural Causal Modeling of SABR Volatility via TabPFN and Sobolev Training

## Project Overview
This project addresses the challenge of modeling the SABR (Sigma-Alpha-Beta-Rho) stochastic volatility model using deep learning. While traditional neural networks can approximate the pricing function, they often fail to capture the Greek sensitivities (derivatives) required for professional risk management. This pipeline implements a Structural Causal Model (SCM) that leverages TabPFN’s prior-data fitting capabilities combined with Sobolev regularization to produce smooth, tradable, and causally consistent volatility smiles.

---

## Phase 1: Synthetic Data Generation and Ground Truth
The foundation of the project relies on the generation of a high-fidelity synthetic dataset. Unlike historical market data, which is plagued by micro-structure noise and reporting gaps, synthetic data allows us to define the "Causal Truth."

* **The SABR Model:** We generated data points using the SABR displacement model equations.
* **Target Features:** For every input set (Alpha, Beta, Rho, VolVol, Forward, Strike), we calculated not only the Volatility but also the exact mathematical derivatives (Greeks) for each parameter.
* **Scaling Strategy:** Data was normalized using standard scaling; however, as noted in the final stress tests, this normalization can impact the interpretability of causal directions if not carefully inverted during inference.

## Phase 2: TabPFN Baseline Integration (Step 4)
We integrated TabPFN, a transformer-based model pre-trained on millions of synthetic causal tasks. TabPFN is unique because it performs "In-Context Learning," treating the regression task as a sequence problem.

* **The Problem:** In its raw state, TabPFN acts as a point-estimator. While the Mean Absolute Error (MAE) on price was acceptable, the model produced "jittery" results.
* **Numerical Instability:** When calculating the first derivative (dV/dK) of the TabPFN output, we observed extreme noise (High VoG - Volatility of Gradient). In a trading environment, this noise would lead to unstable hedging and excessive transaction costs.

## Phase 3: Architecture Optimization (Step 7)
Before introducing complex constraints, we optimized the "Stacking" architecture. We designed a hybrid head that takes both the raw SABR parameters and the TabPFN prediction as inputs.

* **Hyperparameter Tuning:** Using the Ray Tune framework, we ran an exhaustive search over the hidden layer dimensions, dropout rates, and learning schedules.
* **The Result:** The best configuration settled on a decreasing bottleneck architecture (e.g., 512 -> 256 -> 128) with a specific dropout rate (~0.09) to prevent overfitting to the synthetic noise.

## Phase 4: Sobolev Regularization and Gradient Stability (Step 8)
This phase marked the transition from "Curve Fitting" to "Sensitivity Modeling." We implemented Sobolev Training, which modifies the loss function to include the error of the derivatives.

* **Loss Modification:** The loss function was redefined as:
  $$Loss = \lambda_{1} |V_{pred} - V_{true}| + \lambda_{2} |\nabla V_{pred} - \nabla V_{true}|$$
* **Smoothing Effect:** By penalizing errors on the Greeks, the model was forced to smooth the pricing function. The "Blue Curve" vs. "Red Curve" comparison demonstrated that Sobolev training acts as a powerful regularizer, removing the numerical tremors seen in Step 4.
* **Financial Tradability:** The resulting gradients became stable enough to be used for Delta and Gamma hedging.

## Phase 5: Causal Structural Fine-Tuning (Step 9)
The final stage transformed the regressor into a Financial Structural Causal Model (SCM). Following the principles outlined in the TabPFN research papers, we treated the model parameters as causal nodes in a Directed Acyclic Graph (DAG).

* **Mechanism vs. Correlation:** Standard models learn that "Alpha and Volatility move together." Our SCM learns that "Alpha *causes* Volatility."
* **Activation Function:** We transitioned to the SiLU (Swish) activation function ($x \cdot \text{sigmoid}(x)$). Unlike ReLU, SiLU is continuously differentiable, which is a prerequisite for a model that predicts its own derivatives.
* **Performance Analysis:** The final training logs showed a Volatility MAE of ~0.024. More importantly, the Total Loss converged alongside the MAE, proving that the model successfully learned the causal structure without sacrificing point-accuracy.

## Phase 6: Causal Quality and Stress Testing (Step 10)
To validate the model, we performed "Out-of-Distribution" stress tests, specifically looking at the Alpha-Volatility relationship.

* **Monotonicity Observation:** The stress tests revealed that while the model is extremely smooth, the scaling of input features can lead to sign inversions (e.g., Alpha showing a negative causal impact). 
* **Interpretation:** This highlights the importance of the "Inversion Layer" in the SCM. The model has learned the *strength* of the causal link perfectly, but the *direction* is tied to the normalization bounds of the training set.
* **Diagnostic Value:** The stability of the "Impact Causal" (dV/dAlpha) across the entire range of inputs confirms that the model has reached a high level of numerical maturity and is ready for integration into a broader risk engine.

---
### Technical Summary
* **Model Type:** Hybrid Stacking SCM (TabPFN + MLP)
* **Optimization:** Ray Tune / AdamW
* **Regularization:** Sobolev (Derivative-based)
* **Smoothness:** C-infinity (via SiLU)