# House Price Prediction using Linear Regression

## Project Overview
This project predicts house prices using Machine Learning with Python. It uses the Linear Regression algorithm to train a model and predict house prices based on housing features.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
- `train.csv` – Training dataset
- `test.csv` – Testing dataset
- `data_description.txt` – Description of dataset columns

## Project Steps
1. Load the dataset using Pandas.
2. Explore dataset information and columns.
3. Identify missing values.
4. Separate input features and output labels.
5. Split data into training and testing sets.
6. Train a Linear Regression model.
7. Make house price predictions.
8. Evaluate model performance.

## Model Evaluation Results

| Metric | Result |
|---|---:|
| Mean Absolute Error | 20466.13 |
| Mean Squared Error | 979376460.09 |
| Root Mean Squared Error | 31294.99 |
| R² Score | 0.8723 |

## Output
The model achieved an R² score of 87.23% on the testing split.

## How to Run

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run the Python file:

```bash
python app.py
```

## Author
santhoshini078
