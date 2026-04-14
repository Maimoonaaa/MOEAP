# One-time setup
git clone https://github.com/YOUR_USERNAME/pet_moeap
cd pet_moeap
pip install -r requirements.txt

# Run full pipeline
python -m data.generate_dataset          # ~5 min for 10k images
python -m models.cnn_objectives          # ~20 min on CPU, ~5 min on GPU
python -m experiments.run_experiment     # ~30 min on CPU

# Results appear in results/