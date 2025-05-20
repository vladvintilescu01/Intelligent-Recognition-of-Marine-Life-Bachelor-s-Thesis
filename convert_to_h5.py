from tensorflow.keras.models import model_from_json

# === Setări fișiere ===
model_name = "Fish_Dataset_Split_DenseNet121"  # fără extensii
base_path = "All weights and structure from experiments/"  # schimbă dacă ai alt folder

json_path = f"{base_path}{model_name}.json"
weights_path = f"{base_path}{model_name}.weights.h5"
output_path = f"{base_path}{model_name}_full.keras"

# === Încarcă arhitectura ===
with open(json_path, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# === Încarcă weights ===
model.load_weights(weights_path)

# === Salvează modelul complet într-un singur fișier .h5 ===
model.save(output_path)

print(f"Model complet salvat ca: {output_path}")
