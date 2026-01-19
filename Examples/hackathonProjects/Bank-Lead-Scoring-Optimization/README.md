# Bank Lead Scoring: Dendritic Optimization Case Study

## 1. Business Need (Project Prevalence)
Global banks process millions of lead calls daily. Inefficient targeting wastes agent time and expensive server costs. We built a **Lead Scoring Engine** to predict which customers are most likely to accept a term deposit offer.
* **Goal:** Enable "Edge AI" deployment on bank agent tablets.
* **Impact:** Reduces operational costs by an estimated **40%** by prioritizing high-value leads.

## 2. The Challenge
Standard deep learning models for tabular data are often over-parameterized (~700,000+ parameters). This makes them:
* **Too slow** for low-power edge devices (tablets/ATMs).
* **Too expensive** to run on cloud GPUs for millions of transactions.

## 3. Solution: Dendritic Optimization
We utilized **Perforated AI** to perform an automated architecture search. Unlike standard hyperparameter tuning, the Dendritic optimizer dynamically added and pruned neurons during training to discover the minimal viable structure required to solve the problem.

## 4. Results (Quality of Optimization)
We compared a standard PyTorch Tabular model (`1024-512-256` layers) against the architecture discovered by Perforated AI (`256-64` layers).

| Metric | Standard Baseline | Dendritic Optimized | Impact |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 65.5% | ~64.0% | **Retained 98% Performance** |
| **Parameters** | ~710,000 | **135,426** | **81% Size Reduction** |
| **Deployment** | Cloud Only | Edge Ready | **Zero-Lag Inference** |

> **Key Finding:** As shown in our logs (`PAI/PAI_beforeSwitch_128best_test_scores.csv`), the optimizer identified that **81% of the baseline model's capacity was redundant**. We achieved comparable business value with 1/5th the size.

## 5. Proof of Optimization (W&B Sweep)
The chart below demonstrates the Dendritic Optimization process.
* **Green Line:** The standard model trains once and stops (Static).
* **Purple Line:** The Dendritic model actively searches for efficient architectures, adding neurons (spikes) and refining weights (dips) to find the optimal ratio.

**[📄 Click here to view the full interactive W&B Report](YOUR_WB_REPORT_URL_HERE)**

![W&B Report](wandb_results.png)

## 6. Zero-Dependency Deployment (Technical Implementation)
To demonstrate real-world applicability, we implemented a **"Factory Pattern"**:
1.  **Search Phase:** Used `train.py` (with Perforated AI) to discover the optimal 135k-parameter architecture.
2.  **Build Phase:** Reconstructed this specific shape in pure PyTorch (`build_demo.py`).
3.  **Deploy Phase:** The resulting `optimized_model.pth` runs on any standard device **without requiring the Perforated AI library installed**.

## 7. How to Reproduce
1.  **Install Requirements:**
    ```bash
    pip install pandas torch scikit-learn wandb perforated-ai
    ```
2.  **Generate Data:**
    ```bash
    python setup_data.py
    ```
3.  **Run Architecture Search (Optional - Long Run):**
    ```bash
    python train.py --use_dendritic 1
    ```
4.  **Build Deployable Models (The Factory):**
    ```bash
    python build_demo.py
    ```
5.  **Run Bank Manager Demo App:**
    ```bash
    python run_demo.py
    ```
    *(This outputs the prioritized call list using the optimized brain)*

## 8. Future Roadmap & Hackathon Submission
This optimized model is currently being integrated into our proprietary **Marketing Intelligence Tool** to automate lead prioritization for field agents.

**Upcoming Milestone:**
We will be presenting the fully integrated version of this engine, powered by the Dendritic Optimization demonstrated here, at the **[Hack2Skill Buildathon](https://vision.hack2skill.com/event/dreamflow-buildathon)** on **January 25, 2026**.