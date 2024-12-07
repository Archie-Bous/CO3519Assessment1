import cv2
import numpy as np
import seaborn as sns
import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import local_binary_pattern
import joblib

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

radius = 3
n_points = 8 * radius
# Function to load images, detect faces, and extract cropped face regions
def load_and_detect_faces(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        face_region = img[y:y+h, x:x+w]
                        images.append(face_region)
                        labels.append(label)
    return images, labels

# Function to extract HOG features from the cropped face images
def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Resize face image to a standard size if needed
        img = cv2.resize(img, (256, 256))  # Standardize size
        # Extract HOG features
        features, hog_image = hog(
            img,
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            block_norm='L2-Hys', 
            visualize=True
        )
        hog_features.append(features)
    return np.array(hog_features)

def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        img = cv2.resize(img, (256,256))
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)
    return np.array(lbp_features)

# Function to preprocess, detect face, extract HOG features, and predict emotion
def predict_emotion_hog(image_path, model):
    plt.cla()
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or could not be loaded.")

    # Detect face in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Assume first detected face (for single face case)
    (x, y, w, h) = faces[0]
    face_region = img[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (256, 256))
    # Extract HOG features from the detected face region
    hog_features = extract_hog_features([face_region])[0]
    #hog_features = hog_features.reshape(1, -1)

    lbp_features = extract_lbp_features([face_region])[0]
    #lbp_features = lbp_features.reshape(1, -1)

    combined_features = np.hstack([hog_features, lbp_features])
    combined_features = combined_features.reshape(1, -1)
    scaler = joblib.load('scaler.pkl')
    scaled_features = scaler.transform(combined_features)

    # Reshape features for model input
    #features = features.reshape(1, -1)

    # Predict the expression using the model
    predicted_expression = model.predict(scaled_features)[0]

    # Load the original image in color for visualization
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Draw a rectangle around the detected face in the color image
    img_with_box = img_rgb.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Red rectangle
    cv2.putText(img_with_box, predicted_expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display three subplots: prediction, face with bounding box, and HOG features
    plt.figure(figsize=(18, 6))

    # 1st subplot: Original image with predicted expression label
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Input Image")
    plt.axis("off")

    # 2nd subplot: Face detected with bounding box
    plt.subplot(1, 3, 2)
    plt.imshow(img_with_box)
    plt.title(f"Predicted Expression: {predicted_expression}")
    plt.axis("off")

    # 3rd subplot: HOG features
    #plt.subplot(1, 3, 3)
    #plt.imshow(hog_image, cmap='gray')
    #plt.title("HOG Features")
    #plt.axis("off")

    plt.show()

# Set paths for training and testing folders
#train_folder_path = "JAFFE-[70,30]\\JAFFE-[70,30]\\train"
#test_folder_path = "JAFFE-[70,30]\\JAFFE-[70,30]\\test"
train_folder_path = "JAFFE-[70,30]/train"
test_folder_path = "JAFFE-[70,30]/test"


print("training ...\n")
# Load, detect faces, and extract HOG features for training data
scaler = StandardScaler()
X_train_faces, y_train = load_and_detect_faces(train_folder_path)
hog_train_features = extract_hog_features(X_train_faces)
lbp_train_features = extract_lbp_features(X_train_faces)
combined_train_features = np.hstack([hog_train_features,lbp_train_features])
X_train_combined_features = scaler.fit_transform(combined_train_features)
joblib.dump(scaler, 'scaler.pkl')

print("testing ...\n")
# Load, detect faces, and extract HOG features for testing data
X_test_faces, y_test = load_and_detect_faces(test_folder_path)
hog_test_features = extract_hog_features(X_test_faces)
lbp_test_features = extract_lbp_features(X_test_faces)
combined_test_features = np.hstack([hog_test_features,lbp_test_features])
X_test_combined_features = scaler.transform(combined_test_features)

# create SVM pipeline for scaling and rbf kernel
svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=5, gamma='scale'))
svm_classifier.fit(X_train_combined_features, y_train)
y_test_pred = svm_classifier.predict(X_test_combined_features)


# accuracy score of training
y_train_pred = svm_classifier.predict(X_train_combined_features)
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# accuracy score of testing
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

# Display Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Assuming class names are defined
class_names = ["Angry", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]

# Plot confusion matrix with class names
plt.figure(figsize=(8, 8))
ax = sns.heatmap(conf_matrix, annot=True, cmap="BuGn", xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
#plt.show()

# Calculate accuracy for each emotion
emotion_accuracies = {}
for i, emotion in enumerate(class_names):
    # True positives for the current emotion
    tp = conf_matrix[i, i]
    # Total instances of the current emotion in the true labels
    total_true = np.sum(conf_matrix[i, :])
    # Accuracy for the current emotion
    accuracy_emotion = tp / total_true if total_true > 0 else 0
    emotion_accuracies[emotion] = accuracy_emotion * 100

# Print accuracy by emotion
print("\nAccuracy by Emotion:")
for emotion, acc in emotion_accuracies.items():
    print(f"{emotion}: {acc:.2f}%")
# Print classification report
print(classification_report(y_test, y_test_pred))

for i in range(5):
    pass
    #image_path = f"C:\\Users\Archie\\OneDrive - UCLan\\Desktop\\Documents\\Uni stuff\\Year4\\Artificial intelligence\\week6\\lab work\\CK_dataset\\CK_dataset\\train\Anger\\{18+i}.jpg"
    #predict_emotion_hog(image_path, svm_classifier)