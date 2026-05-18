import tensorflow as tf
import os

# Disable oneDNN to avoid optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model (without compiling to avoid hardware mismatches)
model = tf.keras.models.load_model('best_traffic_sign_model.keras', compile=False)

# Recompile with standard CPU‑friendly settings
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save as a new file (native Keras format)
model.save('best_traffic_sign_model_fixed.keras')
print("Model converted and saved as 'best_traffic_sign_model_fixed.keras'")

