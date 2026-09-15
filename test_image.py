from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle
import os

# Load the model and label binarizer
model_path = "model.h5"
label_path = "lb.pickle"

print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully!")

print("Loading label binarizer...")
with open(label_path, "rb") as f:
    lb = pickle.load(f)
print("Label binarizer loaded successfully!")
print("Classes:", lb.classes_)

# Image preprocessing parameters
mean = np.array([123.68, 116.779, 103.939], dtype="float32")

def is_valid_image(image):
    """Simple sanity checks to reject invalid images"""
    # Check if image is too dark (mean < 30)
    if np.mean(image) < 30:
        return False, "Image too dark"
    
    # Check if image is too bright (mean > 225)
    if np.mean(image) > 225:
        return False, "Image too bright"
    
    # Check if image has very low variance (solid color)
    if np.var(image) < 100:
        return False, "Image appears to be solid color"
    
    return True, "Valid image"

def predict_image(image_path):
    """Predict the disease class for a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return None
    
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'!")
        return None
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform sanity checks
    is_valid, error_msg = is_valid_image(image)
    if not is_valid:
        print(f"Invalid image: {error_msg}")
        return None
    
    # Resize to 224x224 (ResNet50 input size)
    image = cv2.resize(image, (224, 224))
    
    image = image.astype("float32")
    
    
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    
    
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    predicted_class = lb.classes_[class_idx]
    
    return predicted_class, confidence, predictions[0]

def test_sample_images():
    """Test the model on sample images from the dataset"""
    print("\n" + "="*50)
    print("Testing model on sample images from dataset...")
    print("="*50)
    
    # Test one image from each class
    classes = ["Fusarium Head Blight", "Healthy Wheat", "Leaf Rust", "Tan Spot"]
    
    for class_name in classes:
        dataset_path = f"Dataset/{class_name}"
        if os.path.exists(dataset_path):
            # Get the first image from each class
            image_files = [f for f in os.listdir(dataset_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
            
            if image_files:
                test_image = os.path.join(dataset_path, image_files[0])
                print(f"\nTesting image from {class_name}:")
                print(f"Image: {test_image}")
                
                result = predict_image(test_image)
                if result:
                    predicted_class, confidence, all_predictions = result
                    print(f"Predicted: {predicted_class}")
                    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                    print(f"Correct: {'✓' if predicted_class == class_name else '✗'}")
                    
                    # Show all class probabilities
                    print("All predictions:")
                    for i, class_name_pred in enumerate(lb.classes_):
                        print(f"  {class_name_pred}: {all_predictions[i]:.4f} ({all_predictions[i]*100:.2f}%)")
        else:
            print(f"Warning: Dataset folder '{dataset_path}' not found!")

if __name__ == "__main__":
    # Test on sample images
    #test_sample_images()
    
    
    result = predict_image(r"C:\Users\yspat\Desktop\PBL\Wheat-ripening-17-july-688.jpg")
    if result:
        predicted_class, confidence, all_predictions = result
        print(f"Predicted: {predicted_class} (Confidence: {confidence:.4f})")
