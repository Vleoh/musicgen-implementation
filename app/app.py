from flask import Flask, request, render_template, send_file
from audiocraft.models import MusicGen
import os
import soundfile as sf
from pydub import AudioSegment

app = Flask(__name__)

# Directorio para guardar archivos generados
OUTPUT_DIR = "static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar el modelo MusicGen
print("Cargando modelo MusicGen...")
model = MusicGen.get_pretrained("melody")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Obtener descripción del usuario
        description = request.form.get("description", "")
        if not description:
            return render_template("index.html", error="Por favor ingresa una descripción.")

        # Generar música con MusicGen
        print(f"Generando música para: '{description}'")
        print(f"Description: {description}")
        generated_audio = model.generate([description])  

        # Guardar el audio generado
        output_path = os.path.join(OUTPUT_DIR, "generated_audio.wav")
        data = generated_audio[0].cpu().numpy().T  # Convertir y transponer los datos del tensor
        sf.write(output_path, data, 32000)  # Guardar el archivo con una frecuencia de muestreo de 32 kHz

        return render_template("index.html", audio_file=output_path)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    # Ruta completa del archivo
    file_path = os.path.join(OUTPUT_DIR, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
