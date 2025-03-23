import pandas as pd
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Tải dữ liệu
df = pd.read_csv('sentimentdataset.csv')

# Hiển thị thông tin dữ liệu
print(df.info())
print(df.head())

# Loại bỏ giá trị thiếu
df = df.dropna(subset=['Text', 'Sentiment']).reset_index(drop=True)

# Tải stopwords từ NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'http\S+|www\S+', '', text)  # Loại bỏ URL
    text = re.sub(r'[^a-z\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Loại bỏ stopwords
    return text

# Áp dụng tiền xử lý
df['Clean_text'] = df['Text'].apply(preprocess_text)

# Chuyển đổi văn bản thành vector TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Clean_text']).toarray()

# Mã hóa nhãn cảm xúc
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Lấy lại chỉ mục của tập kiểm tra
test_indices = df.index[len(X_train):]  # Chỉ mục của tập kiểm tra

# Khởi tạo mô hình Naïve Bayes với Laplace Smoothing
nb_model = MultinomialNB(alpha=1.0)

# Huấn luyện mô hình
nb_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = nb_model.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Báo cáo chi tiết
print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=le.classes_))

# -------------------------------------------
# Hàm dự đoán cảm xúc của một câu mới
# -------------------------------------------
def predict_sentiment(text):
    clean_text = preprocess_text(text)  # Tiền xử lý câu mới
    vectorized_text = tfidf.transform([clean_text]).toarray()  # Vector hóa
    prediction = nb_model.predict(vectorized_text)  # Dự đoán
    return le.inverse_transform(prediction)[0]  # Giải mã nhãn

# Test thử với một câu bất kỳ
test_sentence = "A surprise gift from a friend made my day!"
predicted_sentiment = predict_sentiment(test_sentence)
print(f"Sentiment Prediction: {predicted_sentiment}")

# -------------------------------------------
# Lưu kết quả kiểm tra ra file CSV
# -------------------------------------------
df_test_results = pd.DataFrame({
    'Text': df.loc[test_indices, 'Text'].values,
    'Actual Sentiment': le.inverse_transform(y_test),
    'Predicted Sentiment': le.inverse_transform(y_pred)
})

df_test_results.to_csv('test_results.csv', index=False)
print("Results saved to test_results.csv")
