# File di input
conda_file = "conda_requirements.txt"
pip_file = "pip_requirements.txt"
output_file = "final_requirements.txt"

# Leggi i pacchetti Conda
with open(conda_file, "r") as f:
    conda_packages = {line.strip().split('=')[0] for line in f if line.strip() and not line.startswith("#")}

# Leggi i pacchetti pip
with open(pip_file, "r") as f:
    pip_packages = [line.strip() for line in f if line.strip()]

# Filtra i pacchetti pip per evitare duplicati con Conda
final_packages = [pkg for pkg in pip_packages if pkg.split("==")[0] not in conda_packages]

# Scrivi il file finale
with open(output_file, "w") as f:
    f.writelines("\n".join(final_packages))

print(f"File requirements combinato generato: {output_file}")


