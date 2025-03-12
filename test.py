import tensorflow as tf

# Load the model from the .keras file
model = tf.keras.models.load_model("ann_model1.keras")

# Save the model in the newer .keras format
model.save("ann_model1_new.keras")

print("Model saved in .keras format as ann_model1_new.keras")