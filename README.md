# Students Performance Predictor

Students Performance Predictor is a machine learning project that predicts a student's academic performance based on various input features such as gender, study time, parental education, and test preparation. It uses a trained regression/classification model for prediction.

## Technologies Used
- Python
- scikit-learn
- Pandas
- NumPy
- Pickle (for model serialization)

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the prediction script:
   ```bash
   python predict_student.py
   ```

3. (Optional) Integrate with a frontend to accept inputs and display results.

## Files Included
- `predict_student.py` – script to load model and predict
- `Models/` – contains trained model (`StudentModel.pkl`), scaler, and encoders
- `requirements.txt` – list of dependencies

