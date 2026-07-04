🛡️ Phishing Link Detection

A machine learning model that classifies URLs as phishing or legitimate, built with LightGBM.

📊 Results


93.2% accuracy on the test set
Gradient-boosted decision tree model (LightGBM) trained on URL/lexical features


🛠️ Tech Stack


Python
LightGBM
Scikit-learn
Pandas / NumPy


📁 What's Inside


Feature extraction from URLs (lexical, host-based, and/or content features — edit to match your actual pipeline)
Model training and evaluation notebook/scripts
Saved model artifact for inference


🚀 Getting Started

bashgit clone https://github.com/janvi2741/Phishing_link_detection.git
cd Phishing_link_detection
pip install -r requirements.txt

Then run the training/inference script — add the exact command, e.g.:

bashpython train.py

📈 Why LightGBM

Briefly explain your choice here — e.g. handles tabular/lexical features well, fast training, good accuracy-to-latency tradeoff for real-time link scanning.

📌 Future Improvements


Real-time browser extension integration
Expanded feature set (WHOIS, SSL cert age, redirect chains)
Model comparison against XGBoost/Random Forest baselines



Built as part of my ML engineering work — flagged as a resume highlight (93.2% accuracy)
