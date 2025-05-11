import pandas as pd
from typing import List, Tuple

def load_cities_from_csv(file_path: str) -> List[Tuple[str, float, float]]:
    df = pd.read_csv(file_path)
    required_columns = ['city', 'latitude', 'longitude']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    return list(zip(df['city'], df['latitude'], df['longitude']))

def load_cities_from_session(session_cities) -> List[Tuple[str, float, float]]:
    return [(city['name'], city['lat'], city['lon']) for city in session_cities] 