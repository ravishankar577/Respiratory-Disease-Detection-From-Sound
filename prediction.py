import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn import preprocessing
from xgboost import XGBClassifier
import antropy as ant

inputs = []
SAMPLE_RATE = 16000
filepath = "169_1b2_Ll_sc_Meditron.wav"
raw, sr = librosa.load(filepath, sr = SAMPLE_RATE)
zcr = librosa.core.zero_crossings(raw).sum() / len(raw)
sc = librosa.feature.spectral_centroid(y=raw)[0]
rms = librosa.feature.rms(y=raw)[0]
s_rf = librosa.feature.spectral_rolloff(y=raw, roll_percent=0.85)[0]
s_rf_75 = librosa.feature.spectral_rolloff(y=raw, roll_percent=0.75)[0]
sf = librosa.feature.spectral_flatness(y=raw)[0]
se = ant.spectral_entropy(x = raw, sf = sr, method='fft')
mfccs = librosa.feature.mfcc(y=raw, hop_length=len(raw), n_mfcc=8)
mfccs = mfccs.flatten()

add_to_array=[
             zcr,
             sc.mean(),
             np.median(sc),
             sc.std(),
             rms.mean(),
             np.median(rms),
             rms.std(), 
             s_rf.mean(), 
             np.median(s_rf),
             s_rf.std(),
             s_rf_75.mean(), 
             np.median(s_rf_75),
             s_rf_75.std(),
            sf.mean(),
            np.median(sf),
            sf.std(),
            se]
add_to_array.extend(mfccs)

inputs.append(add_to_array)

# add_to_array_ = preprocessing.scale(add_to_array)

data_columns = [
    "zero_crossing_rate", 
                "spectral_centroid_mean", 
                "spectral_centroid_median",
                "spectral_centroid_std", 
                "root_mean_square_mean", 
                "root_mean_square_median", 
                "root_mean_square_std", 
                "spectral_rolloff_85_mean", 
                "spectral_rolloff_85_median", 
                "spectral_rolloff_85_std",
               "spectral_rolloff_75_mean", 
                "spectral_rolloff_75_median", 
                "spectral_rolloff_75_std",
               "spectral_flatness_mean",
               "spectral_flatness_median",
               "spectral_flatness_std",
               "spectral_entropy",
               "mfcc1",
               "mfcc2",
               "mfcc3",
               "mfcc4",
               "mfcc5",
               "mfcc6",
               "mfcc7",
               "mfcc8",
               "mfcc9",
               "mfcc10",
               "mfcc11",
               "mfcc12",
               "mfcc13",
               "mfcc14",
               "mfcc15",
               "mfcc16"]
np_breathing_data_array = np.array(inputs)
np_breathing_data_array.shape
breathing_data_df = pd.DataFrame(np_breathing_data_array, columns=data_columns)
model = XGBClassifier()
model.load_model("model.json")
preds = model.predict(breathing_data_df)

if preds[0] == 0:
    prediction = "Bronchiectasis"
elif preds[0] == 1:
    prediction = "Bronchiolitis"
elif preds[0] == 2:
    prediction = "COPD"
elif preds[0] == 3:
    prediction = "Healthy"
elif preds[0] == 4:
    prediction = "Pneumonia"
elif preds[0] == 5:
    prediction = "URTI"
    
print("============================================================================")
print("preds",preds)
print("Prediction is: ",prediction)