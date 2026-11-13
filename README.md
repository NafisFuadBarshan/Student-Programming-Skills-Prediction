# Student Programming Skills Prediction

Predicting how many programming languages a student knows using **multivariate linear regression**, implemented two ways: from scratch with **NumPy** (Normal Equation) and with **scikit-learn**. The two models are then compared on accuracy, training time, and learned coefficients.

> Course project for **CSE 404: Artificial Intelligence & Expert Systems Lab**, Department of Computer Science & Engineering, University of Asia Pacific.

---

## Overview

| | |
|---|---|
| **Task** | Regression: estimate a student's number of programming languages known |
| **Target** | `How many programming languages do you know?` |
| **Features** | 24 academic, technical, and personal attributes |
| **Dataset** | `Studentdata.csv` (254 student records, 25 columns) |
| **Models** | Manual NumPy implementation vs. `sklearn.linear_model.LinearRegression` |
| **Split** | 80% train / 20% test (`random_state=42`) |

## Results

Both models produce **identical** results, which confirms the manual implementation is correct.

| Metric | Scikit-learn | Manual (NumPy) |
|---|---|---|
| MSE | 1.6366 | 1.6366 |
| RMSE | 1.2793 | 1.2793 |
| MAE | 1.0118 | 1.0118 |
| R² | 0.2831 | 0.2831 |
| Training time (s) | 0.00331 | 0.000378 |

The R² of about 0.28 means a linear model on these label-encoded survey features explains only a modest share of the variance. See [Limitations](#limitations).

## Repository Structure

```
.
├── Multivariate_Linear_Regression_on_StudentData.ipynb   # Main notebook (full pipeline)
├── Studentdata.csv                                       # Dataset
├── Training Evaluation Metrics Table.xlsx                # Metrics comparison table
├── Technical Report.pdf                                  # Full written report
├── Presentation.pptx                                     # Project slides
├── Output Visuals/
│   ├── 1.png                                             # Metrics comparison bar chart
│   └── 2.png                                             # Actual vs. predicted scatter plots
└── README.md
```

## Methodology

1. **Load data** from `Studentdata.csv` with pandas.
2. **Clean column names** by stripping numbering and whitespace (e.g. `"1. How many programming languages do you know? "` becomes `"How many programming languages do you know?"`).
3. **Map the target** to a numeric scale:

   | Survey answer | Value |
   |---|---|
   | None | 0 |
   | 1 | 1 |
   | 2–3 | 2.5 |
   | 4–5 | 4.5 |
   | More than 5 | 6 |

4. **Drop rows** whose target is missing after mapping.
5. **Encode categorical features** with `LabelEncoder`.
6. **Split** into training and test sets (80/20).
7. **Train both models** and evaluate with MSE, RMSE, MAE, and R².
8. **Compare** metrics, training time, and coefficients, and plot the results.

### Manual implementation

The from-scratch model solves for the weights directly with the **Normal Equation**:

```
θ = (XᵀX)⁻¹ Xᵀ y
```

A bias column of ones is prepended to the feature matrix. Because the solution is analytical and non-iterative, there is no learning rate or iteration count to tune.

```python
class ManualLinearRegression:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)
```

## Dataset

Survey responses from 254 students, grouped roughly as:

- **Personal and demographic:** gender, year of study, branch, first-generation graduate status
- **Academic history:** schooling type, percentage, and school location for Class 10 and Class 12; gap year
- **Technical skills and engagement:** active technologies, DSA exposure, GitHub exposure, developer communities, hackathons, software and hardware projects, idea pitching
- **Extracurricular:** volunteer groups, additional skills, achievements

## Getting Started

### Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- Jupyter or Google Colab

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Run on Google Colab (as originally developed)

1. Open `Multivariate_Linear_Regression_on_StudentData.ipynb` in Colab.
2. Run the cells and upload `Studentdata.csv` when the file picker appears.

### Run locally

The notebook uses `google.colab.files.upload()` for loading data. To run it locally, remove or comment out these lines:

```python
from google.colab import files
uploaded = files.upload()
```

and make sure `Studentdata.csv` is in the same directory as the notebook. Then:

```bash
jupyter notebook Multivariate_Linear_Regression_on_StudentData.ipynb
```

## Limitations

- **Label encoding** imposes an arbitrary numeric order on categorical features, which a linear model treats as meaningful. One-hot encoding would likely be more appropriate for nominal features.
- **The target is binned.** Survey ranges were converted to approximate midpoints, so predictions are only as precise as those approximations.
- **Small dataset.** 254 records (about 200 for training) limits how well the model can generalize.
- **Modest fit.** An R² of about 0.28 suggests that nonlinear models or better feature engineering could improve results.

## Possible Improvements

- One-hot encode nominal features and ordinally encode ordered ones
- Add feature scaling, regularization (Ridge/Lasso), and cross-validation
- Try tree-based models or treat the target as an ordinal classification problem
- Report feature importance from the learned coefficients

## Authors

| Name | Reg. No. |
|---|---|
| Nafis Fuad Barshan | 21201166 |
| Tanvir Rahan Rifat | 22101065 |
| Md Adnan Asif | 22101081 |

**Course:** CSE 404, Section B
**Instructor:** Nahida Marzan, Lecturer, Department of CSE, University of Asia Pacific

## License

No license has been specified. Add one (e.g. MIT) if you plan to share or reuse this work publicly.
