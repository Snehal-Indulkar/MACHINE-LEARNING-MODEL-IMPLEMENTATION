# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SNEHAL NAMDEV INDULKAR

*INTERN ID*: CT4MXIT

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 16 WEEKS

*MENTOR*: NEELA SANTOSH

# 📩 SMS Spam Detection using NLP and Machine Learning

## 🧠 Overview

This project is a **machine learning-based SMS Spam Detection system** developed in **Python**. It uses **Natural Language Processing (NLP)** to classify text messages into two categories: **spam** and **ham** (non-spam). The model uses a **Multinomial Naive Bayes** classifier trained on a public SMS dataset and is evaluated using accuracy score and confusion matrix.

It serves as a practical demonstration of using text classification in real-world applications such as email filters, messaging apps, and customer support automation.

---

## 🛠️ Tools & Technologies Used

- **Language**: Python
- **Environment**: Jupyter Notebook 
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation and preprocessing
  - `matplotlib`, `seaborn` – Data visualization
  - `sklearn` – Model training, testing, and evaluation
- **Model**: Multinomial Naive Bayes (from `sklearn.naive_bayes`)
- **Dataset**: [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)

---

## ⚙️ How It Works

1. **Data Loading**  
  - The SMS dataset is loaded directly from a GitHub URL.
  - It includes SMS text messages labeled as either spam or ham.

2. **Data Preprocessing**
   - The text messages are assigned column names (`label`, `message`).
   - Labels are converted to numeric format: `spam` → 1, `ham` → 0.

3. **Text Vectorization**
   - Text messages are converted into numeric form using `CountVectorizer` (Bag-of-Words approach).

4. **Model Training**
   - The dataset is split into training and testing sets using `train_test_split`.
   - A **Multinomial Naive Bayes** model is trained on the training data.

5. **Model Evaluation**
   - The model is tested on unseen data.
   - Metrics such as accuracy, classification report, and confusion matrix are generated.
   - A heatmap of the confusion matrix is visualized using `seaborn`.

---
## ▶️ How to Run the Project

1. **Open the Project in Jupyter Notebook**
   - Launch Jupyter Notebook from Anaconda Navigator or using `jupyter notebook` command.
   - Open the `sms_spam_classifier.ipynb` file.

2. **Install Required Libraries** (if not already installed):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

3. Run All Cells

- Execute each code cell in order to perform data loading, preprocessing, training, and evaluation.

📌 Applications
- Filtering spam SMS messages in mobile devices.

- Email spam detection systems.

- Chatbot and messaging moderation.

- Automated helpdesk filtering to detect irrelevant or harmful messages.

✅ Key Features
- Real-time spam classification using machine learning.

- Uses real-world SMS dataset.

- Simple implementation and easy to understand.

- Visual analysis using plots.

- Completely executable within Jupyter Notebook.


