import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Constants
DATA_DIR = "data"
SAMPLE_FILE = os.path.join(DATA_DIR, "sample.tif")

# Functions for EO Processing
def load_image(file_path):
    """Load GeoTIFF image using Rasterio."""
    with rasterio.open(file_path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

def display_image(image, title="Satellite Image", bands=(1, 2, 3)):
    """Display a satellite image using matplotlib."""
    rgb_image = np.stack([image[b - 1] for b in bands], axis=-1)
    plt.imshow(rgb_image / np.max(rgb_image))
    plt.title(title)
    plt.axis("off")
    plt.show()

def normalize(image):
    """Normalize image bands."""
    return np.array([(band - np.min(band)) / (np.max(band) - np.min(band)) for band in image])

# Geospatial Analysis
def load_vector_data(file_path):
    """Load vector data (shapefile) using GeoPandas."""
    return gpd.read_file(file_path)

# Machine Learning Model
def train_model(X, y):
    """Train a Random Forest Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return clf

def create_training_data(image):
    """Generate synthetic training data."""
    flat_image = image.reshape(image.shape[0], -1).T
    labels = np.random.randint(0, 2, flat_image.shape[0])  # Example binary labels
    return flat_image, labels

# Main Workflow
if __name__ == "__main__":
    # Load and visualize image
    image, profile = load_image(SAMPLE_FILE)
    print("Image loaded with shape:", image.shape)
    display_image(image, title="Original Satellite Image", bands=(3, 2, 1))  # RGB

    # Normalize and process data
    norm_image = normalize(image)
    print("Image normalized.")

    # Generate synthetic training data and train model
    X, y = create_training_data(norm_image)
    model = train_model(X, y)
    print("Model trained successfully.")
