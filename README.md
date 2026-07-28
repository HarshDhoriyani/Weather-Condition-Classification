# Assignment 6 — Weather Condition Classification using SVM and Open-Meteo API

## Objective
Build a Support Vector Machine (SVM) classifier that predicts whether the weather is
**Cool** or **Warm** based on meteorological observations (temperature, relative humidity,
surface pressure, wind speed) obtained from the Open-Meteo API.

## API Documentation Link
- Open-Meteo Forecast API: https://open-meteo.com/en/docs
- Endpoint used:
  `https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7`
- Location: New Delhi, India (28.6139° N, 77.2090° E)

## Libraries Used
- `requests` — fetching data from the Open-Meteo API
- `pandas`, `numpy` — data handling
- `scikit-learn` — preprocessing, SVM model, evaluation metrics
- `matplotlib`, `seaborn` — visualization (confusion matrix)

## Methodology
1. **Data Collection:** Fetched 7 days (168 hourly records) of temperature, relative humidity,
   surface pressure, and wind speed from the Open-Meteo Forecast API and loaded it into a
   Pandas DataFrame.
2. **Target Creation:** Derived a `Weather_Class` column — `Warm` if `temperature_2m ≥ 25°C`,
   else `Cool`.
3. **Preprocessing:**
   - Verified there were no missing values.
   - Dropped the non-predictive `time` column.
   - Label-encoded the target (`Cool → 0`, `Warm → 1`).
   - Split data 80/20 into train/test sets (stratified).
   - Standardized all four features with `StandardScaler`.
4. **Model Development:** Trained an `SVC(kernel="rbf")` classifier on the scaled training data
   and generated predictions on the test set.
5. **Evaluation:** Computed accuracy, precision, recall, F1-score, and a confusion matrix.

## Results
| Metric | Score |
|---|---|
| Accuracy | 0.8824 |
| Precision | 1.0000 |
| Recall | 0.8000 |
| F1-Score | 0.8889 |

**Confusion Matrix**

|  | Predicted Cool | Predicted Warm |
|---|---|---|
| **Actual Cool** | 14 | 0 |
| **Actual Warm** | 4 | 16 |

### Observations
1. The model achieved perfect precision on the Warm class — it never misclassified a Cool
   reading as Warm — but recall for Warm was 0.80, meaning 4 borderline Warm samples (close to
   the 25°C threshold) were labeled Cool.
2. All 14 actual Cool samples were correctly classified (recall = 1.00), showing the RBF kernel
   separates the two regimes well away from the boundary.
3. Overall accuracy of 88.2% and F1-score of 0.89 confirm the SVM effectively captured the
   relationship between the four meteorological features and the Warm/Cool label, with most
   errors concentrated near the 25°C cutoff where humidity/pressure add ambiguity.

## Conclusion
This project applied an SVM (RBF kernel) to classify weather conditions as Cool or Warm using
live meteorological data from the Open-Meteo API. The model reached 88.2% accuracy and an
F1-score of 0.89, correctly identifying all Cool samples and most Warm samples, with errors
concentrated near the 25°C decision boundary. Feature scaling was essential because SVM relies
on distance/kernel computations, and features here (temperature, humidity, pressure, wind speed)
have very different numeric ranges — without `StandardScaler`, higher-magnitude features like
pressure could dominate the RBF kernel's distance calculations. A key advantage of SVM is its
ability to model non-linear decision boundaries effectively even on small-to-moderate datasets
using kernel functions. A limitation is that SVM training becomes computationally expensive on
large datasets and its performance depends heavily on hyperparameter tuning (`C`, `gamma`) and
proper feature scaling.

## How to Run
```bash
pip install requests pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Assignment-6.ipynb
```
