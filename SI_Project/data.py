import streamlit as st
import pickle

# Load the model
model = pickle.load(open(r"C:/Parkinsons 2ND SEM THIRD YEAR/New folder/model.pkl", "rb"))

def predict_parkinsons(features):
    # Convert inputs to float
    test = [[float(val) for val in features.values()]]
    # Make prediction
    result = model.predict(test)
    return result[0]

def main():
    st.title('Parkinson\'s Disease Prediction')
    st.write('Enter the values to predict Parkinson\'s disease')

    # Create input fields for each feature
    features = {}
    features['MDVP:Fo(Hz)'] = st.number_input('MDVP: Fo(Hz) : Average vocal fundamental frequency in Hertz.', min_value=70.0, max_value=300.0 )
    features['MDVP:Fhi(Hz)'] = st.number_input('MDVP: Fhi(Hz) : Maximum vocal fundamental frequency in Hertz.', min_value=100.0, max_value=600.0)
    features['MDVP:Flo(Hz)'] = st.number_input('MDVP: Flo(Hz) : Minimum vocal fundamental frequency in Hertz.', min_value=0.0)
    features['MDVP:Jitter(%)'] = st.number_input('MDVP: Jitter(%) : Variation in frequency over time, measured in percentage.', min_value=0.0)
    features['MDVP:Jitter(Abs)'] = st.number_input('MDVP: Jitter(Abs) : Absolute jitter in frequency variation.', min_value=0.0)
    features['MDVP:RAP'] = st.number_input('MDVP: RAP : Relative amplitude perturbation, a measure of variation in amplitude.', min_value=0.0)
    features['MDVP:Shimmer(dB)'] = st.number_input('MDVP: Shimmer(dB) : Variation in amplitude over time, measured in decibels.', min_value=0.0)
    features['Shimmer:DDA'] = st.number_input('Shimmer: DDA : Shimmer in amplitude variation, calculated using the difference between consecutive points.', min_value=0.0)
    features['NHR'] = st.number_input('NHR : Noise-to-harmonics ratio, indicating the ratio of noise to harmonics in the signal.', min_value=0.0)
    features['HNR'] = st.number_input('HNR : Harmonics-to-noise ratio, a measure of the ratio between harmonics and noise in the signal.', min_value=0.0)
    features['RPDE'] = st.number_input('RPDE : Recurrence period density entropy, a measure of the predictability of recurrence intervals.', min_value=0.0)
    features['DFA'] = st.number_input('DFA : Detrended fluctuation analysis, quantifying the fractal scaling properties of time series.', min_value=0.0)
    features['spread1'] = st.number_input('Spread1 : Measure of vocal fold oscillation.', min_value=0.0)
    features['spread2'] = st.number_input('Spread2 : Another measure of vocal fold oscillation.', min_value=0.0)
 

    if st.button('Predict'):
        result = predict_parkinsons(features)
        if result == 0:
            st.write('The person doesn\'t have Parkinson\'s disease')
        else:
            st.write('The person has Parkinson\'s disease')

if __name__ == '__main__':
    main()
