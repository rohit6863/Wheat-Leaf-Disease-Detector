from sklearn.preprocessing import LabelBinarizer
import pickle

# Define the same labels used in training
LABELS = ["Fusarium Head Blight", "Healthy Wheat", "Leaf Rust", "Tan Spot", "Unknown"]

# Create the label binarizer
lb = LabelBinarizer()
lb.fit(LABELS)

# Save the label binarizer
with open("lb.pickle", "wb") as f:
    pickle.dump(lb, f)

print("Label binarizer saved as lb.pickle")
print("Classes:", lb.classes_)
