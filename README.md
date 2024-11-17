# Healthcare-insurance-prediction
# Healthcare Insurance Cost Predictor 🏥

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.0%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-green)

A machine learning solution for predicting healthcare insurance costs based on personal and health-related factors. The project implements multiple regression models and provides an interactive web interface for predictions.

## Features ✨

- Multiple regression models comparison
- Interactive Streamlit web interface
- Comprehensive data preprocessing
- Feature importance analysis
- Model performance visualization
- Automated model selection
- Cross-validation implementation

## Project Structure 📁

```
healthcare-insurance-prediction/
├── healthcare_final.ipynb
└── setup.py
```

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/zainhammagi12/healthcare-insurance-prediction.git
cd healthcare-insurance-prediction
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Performance 📊

### Key Metrics:
- R² Score: 0.7049
- MAE: 3,245.67
- MSE: 26,534,892
- RMSE: 5,151.19

### Model Comparison:
1. Linear Regression: R² = 0.65
2. Random Forest: R² = 0.68
3. CatBoost: R² = 0.7049 (Best Performer)
4. XGBoost: R² = 0.69

## Usage Guide 🚀

### Training Pipeline
```python
from src.pipeline.train_pipeline import TrainPipeline

# Initialize and run training pipeline
train_pipeline = TrainPipeline()
train_pipeline.run_pipeline()
```

### Making Predictions
```python
from src.pipeline.prediction_pipeline import PredictionPipeline

# Initialize predictor
predictor = PredictionPipeline()

# Sample input
input_data = {
    'age': 35,
    'sex': 'male',
    'bmi': 22.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southwest'
}

# Get prediction
prediction = predictor.predict(input_data)
```

### Running Web Interface
```bash
streamlit run streamlit_app.py
```

## Feature Engineering 🔧

### Preprocessing Steps:
1. Handling missing values
2. Encoding categorical variables
3. Feature scaling
4. BMI categorization
5. Age group binning
6. Region encoding

### Feature Importance:
1. Smoking Status (0.42)
2. BMI (0.28)
3. Age (0.15)
4. Region (0.08)
5. Number of Children (0.05)
6. Gender (0.02)

## Web Interface Features 🖥️

- Interactive input form
- Real-time predictions
- Feature importance visualization
- Prediction confidence intervals
- Cost breakdown analysis
- Historical prediction tracking


## Future Improvements 🔮

- [ ] Add deep learning models
- [ ] Implement model monitoring
- [ ] Add feature interaction analysis
- [ ] Create API endpoint
- [ ] Add more visualization options
- [ ] Implement model versioning
- [ ] Add automated retraining

## Contributing 🤝

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

## Author ✍️

Mohammad Hammagi
- LinkedIn: [Zain Hammagi](https://www.linkedin.com/in/zain-hammagi)
- GitHub: [@zainhammagi12](https://github.com/zainhammagi12)


## Acknowledgments 🙏

- Kaggle for the dataset
- Scikit-learn community
- Streamlit documentation
- CatBoost developers
