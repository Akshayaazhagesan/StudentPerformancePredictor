import joblib
import pandas as pd

# Load saved model and tools
model = joblib.load('models/student_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/encoders.pkl')

# Feature names (must match training)
feature_names = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences']

# Sample input (your student data — already encoded)
sample_input = [0, 1, 0, 2, 0, 1, 0, 1, 1, 2,
                3, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 1, 1, 1, 2, 3, 1]

# Convert to DataFrame
sample_df = pd.DataFrame([sample_input], columns=feature_names)

# Scale the input using saved scaler
sample_scaled = scaler.transform(sample_df)

# Make prediction
prediction = model.predict(sample_scaled)

# 🪄 Decode from 0/1 → 'pass' or 'fail'
decoded = label_encoders['performance'].inverse_transform(prediction)

# Show result
print("🎓 Predicted student performance:", decoded[0])
