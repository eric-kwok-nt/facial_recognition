import os
import site
# Resolve version conflict from one of the files
site_packages = site.getsitepackages()
for site in site_packages:
    vggface_model_path = os.path.join(site,'keras_vggface/models.py')
    if os.path.exists(vggface_model_path):
        text = open(vggface_model_path).read()
        open(vggface_model_path, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))
        break