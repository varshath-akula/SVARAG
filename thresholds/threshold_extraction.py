import numpy as np
import pandas as pd
"""
Note : 
These thresholds are utilized in the Sentiment_Analysis_Module to categorize vocal features
into low, medium, and high ranges. This categorization helps in performing Multimodal 
sentiment analysis, where vocal characteristics contribute to determining the emotional 
tone of speech.
"""
def extract_thresholds():
    """
        Extracts thresholds for various audio features from the IEMOCAP dataset.

        This function reads the dataset from "iemocap_audio_features.csv" and computes
        gender-specific thresholds for key audio features. The thresholds are based on
        the 25th percentile (Q1) and the 75th percentile (Q3), categorizing values as
        "low" or "medium_high."

        Features analyzed:
        - duration: Total duration of speech
        - avg_intensity: Average loudness of speech
        - intensity_variation: Variability in loudness
        - avg_pitch: Average pitch frequency
        - pitch_std: Pitch standard deviation
        - pitch_range: Difference between the highest and lowest pitch
        - articulation_rate: Rate of syllables spoken per second
        - mean_hnr: Harmonics-to-noise ratio (voice clarity indicator)

        Returns:
            dict: A nested dictionary with gender-specific thresholds.
                  Example output:
                  {
                      'M': {'avg_pitch': {'low': 120.5, 'medium_high': 190.2}, ...},
                      'F': {'avg_pitch': {'low': 200.3, 'medium_high': 270.1}, ...}
                  }
    """
    df = pd.read_csv("iemocap_audio_features.csv")
    features = ["duration", "avg_intensity", "intensity_variation", "avg_pitch", "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]

    thresholds = {}
    # Calculate gender-specific thresholds, stats
    for gender in ['M', 'F']:
        gender_df = df[df['gender'] == gender] # Filter data by gender
        gender_thresholds = {}
        for feature in features:
            feature_data = gender_df[feature].replace(0, np.nan).dropna() # Ignore 0 and NaN values
            if len(feature_data) > 0:
                q1, q3 = feature_data.quantile([0.25, 0.75]) # Calculate Q1 and Q3
                gender_thresholds[feature] = {'low': q1, 'medium_high': q3}
        thresholds[gender] = gender_thresholds
    return thresholds

if __name__ == "__main__":
    print("Thresholds Extracted from IEMOCAP Dataset : \n")
    print(extract_thresholds())