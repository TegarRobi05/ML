import streamlit as st
import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from wordcloud import WordCloud, STOPWORDS      # ✅ sudah lengkap
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize


import nltk

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# --- Configuration ---
st.set_page_config(
    page_title="Analisis Sentimen Film Jumbo",
    page_icon="🎬",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure it's in the correct directory.")
        return pd.DataFrame()

# Load the initial dataset
df_raw = load_data('film_jumbo.csv')

# Load pre-processed data if available (from your Colab outputs)
# Assuming you saved these files after processing in Colab
df_stemmed = load_data('StemmingJumbo.csv')
df_translated_labeled = load_data('translateJumboo.csv')
df_translated_labeled.columns = df_translated_labeled.columns.str.strip()


# --- Preprocessing Functions (from your Colab notebook) ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text

norm = {
    " yg ": " yang ", " gk ": " tidak ", " ga ": " tidak ", " knp ": " kenapa ",
    " ngga ": " tidak ", " ga ": " tidak ", " gak ": " tidak ", " engga ": " tidak ",
    " enggak ": " tidak ", " nggak ": " tidak ", " enda ": " tidak ", " gua ": " aku ",
    " gue ": " aku ", " gwe ": " aku ", " melek ": " sadar ", " mantap ": " keren ",
    " drpd ": " daripada ", " elu ": " kamu ", " lu ": " kamu ", " lo ": " kamu ",
    " elo ": " kamu ", " nobar ": " nonton bersama ", " krn ": " karena ", " gw ": " aku ",
    " guwe ": " aku ", " ges ": " guys ", " gaes ": " guys ", " kayak ": " seperti ",
    " skrg ": " sekarang ", " taun ": " tahun ", " thh ": " tahun ", " th ": " tahun ",
    " org ": " orang ", " udah ": " sudah ", " kpd ": " kepada ", " gaakan ": " tidak akan ",
    " udh ": " sudah ", " malem ": " malam ", " males ": " malas", " asu ": " anjing ",
    " dg ": " dengan ", " dgn ": " dengan ", " kyk ": " seperti ", " kayaknya ": " sepertinya ",
    " kyaknya ": " sepertinya ", " paslon ": " pasangan calon ", " gaa ": " tidak ",
    " emg ": " emang ", " asep ": " asap ", " bgt ": " banget ", " karna ": " karena ",
    " muuuanis ": " manis ", " pilem ": " film ", " lom ": " belum ", " lbh ": " lebih ",
    " boring ": " bosan ", " bgttttt ": " banget ", " abis ": " habis ", " cuan ": " duit ",
    " jnck ": " jancok ", " jancuk ": " jancok ", " cok ": " jancok ", " jd ": " jadi ",
    " knp ": " kenapa ", " meleduk ": " meledak ", " kgt ": " kaget ", " dpt ": " dapat ",
    " rmhnya ": " rumahnya ", " rmh ": " rumah ", " nntn ": " nonton ", " gla ": " gula ",
    " byk ": " banyak ", " bnyk ": " banyak ", " kmrn ": " kemaren ", " kemarn ": " kemaren ",
    " kmaren ": " kemaren ", " gpp ": " tidak apa apa", " gapapa ": "  tidak apa apa ",
    " uda ": " sudah ", " udh ": " sudah ", " blm ": " belum ", " tp ": " tapi ",
    " gr ": " gara ", " grgr ": " gara gara ", " kocak ": " lucu ", " b aja ": " biasa aja ",
    " b aj ": "  biasa aja ", " gaperlu ": " tidak perlu ", " klean ": " kalean ",
    " aja ": " saja ", " gitu ": " seperti itu ", " nih ": " ini ", " tuh ": " itu ",
    " dmna ": " dimana ", " kyk gitu ": " seperti itu ", " kyk nya ": " sepertinya ",
    " apa gitu ": " apa seperti itu ", " ngapain ": " mengapa ", " nntn ": " nonton ",
    " bs ": " bisa ", " gaes ": " teman-teman ", " trus ": " terus ", " sdh ": " sudah ",
    " dr ": " dari ", " hrs ": " harus ", " misal ": " misalnya ", " mksd ": " maksud ",
    " plg ": " pulang ", " lg ": " lagi ", " gk ": " tidak ", " g ": " tidak ",
    " dah ": " sudah ", " dalem ": " dalam ", " kalo ": " jika ", " trs ": " terus ",
    " ortu ": " orang tua ", " anak2 ": " anak-anak ", " skr ": " sekarang ", " jd ": " jadi ",
    " dgn ": " dengan ", " mgkn ": " mungkin ", " ngaruh ": " berpengaruh ", " skli ": " sekali ",
    " cm ": " cuma ", " gausah ": " tidak usah ", " begtu ": " begitu ", " bnyk bgt ": " sangat banyak ",
    " btw ": " omong-omong ", " apalagi ": " terlebih lagi ", " tpi ": " tapi ",
    " pdhl ": " padahal ", " kyknya ": " sepertinya ", " soalnya ": " karena ", " jg ": " juga ",
    " kmu ": " kamu ", " aku ": " saya ", " ngerasa ": " merasa ", " kagak ": " tidak ",
    " jadiin ": " jadikan ", " gaes ": " teman-teman ", " gaje ": " gak jelas ",
}

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

# Stopword removal setup
factory = StopWordRemoverFactory()
more_stop_words = ['tidak']
stop_words = factory.get_stop_words()
stop_words.extend(more_stop_words)
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

def remove_stopwords(text):
    if not isinstance(text, str):
        return ""
    return stop_words_remover_new.remove(text)

def tokenize_text(text):
    if not isinstance(text, str):
        return []
    return wordpunct_tokenize(text) 

# Stemming (assuming you have a stemmed CSV, otherwise you'd need a stemmer here)
# For demonstration, we'll use the pre-stemmed data if available.
# If not, you'd integrate a stemmer like Sastrawi's StemmerFactory.

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu_selection = st.sidebar.radio(
    "Go to",
    ["Home", "Preprocessing", "Labeling", "Classification", "Model Evaluation"]
)

# --- Home Page ---
if menu_selection == "Home":
    st.title("Welcome to Film Jumbo Sentiment Analysis Dashboard")
    st.write("""
        This dashboard provides an interactive platform to explore sentiment analysis
        of tweets related to the movie 'Film Jumbo'. You can perform various
        text preprocessing steps, view sentiment labels, train a classification model,
        and evaluate its performance.
    """)
    st.header("Original Data Sample")
    if not df_raw.empty:
        st.dataframe(df_raw.head())
        st.write(f"Total data points: {df_raw.shape[0]}")
    else:
        st.warning("Original data not loaded. Please check 'film_jumbo.csv'.")

# --- Preprocessing Page ---
elif menu_selection == "Preprocessing":
    st.title("Text Preprocessing")
    st.write("Perform various text cleaning and normalization steps.")

    df_processed = df_raw.copy()
    if df_processed.empty:
        st.warning("No data to process. Please ensure 'film_jumbo.csv' is loaded.")
    else:
        st.subheader("Original Text")
        st.dataframe(df_processed[['full_text']].head())

        # Cleaning
        if st.checkbox("Apply Cleaning (Remove URLs, Mentions, Hashtags, Punctuation, Numbers)"):
            df_processed['full_text'] = df_processed['full_text'].apply(clean_text)
            df_processed['full_text'] = df_processed['full_text'].str.lower() # Convert to lowercase
            st.subheader("After Cleaning & Lowercasing")
            st.dataframe(df_processed[['full_text']].head())

        # Normalization
        if st.checkbox("Apply Normalization (Typo Correction)"):
            df_processed['full_text'] = df_processed['full_text'].apply(normalisasi)
            st.subheader("After Normalization")
            st.dataframe(df_processed[['full_text']].head())

        # Stopword Removal
        if st.checkbox("Apply Stopword Removal"):
            df_processed['full_text'] = df_processed['full_text'].apply(remove_stopwords)
            st.subheader("After Stopword Removal")
            st.dataframe(df_processed[['full_text']].head())

        # Tokenization
        if st.checkbox("Apply Tokenization"):
            df_processed['tokenized_text'] = df_processed['full_text'].apply(tokenize_text)
            st.subheader("After Tokenization")
            st.dataframe(df_processed[['tokenized_text']].head())

        # Stemming (using pre-stemmed data if available, otherwise show a note)
        st.subheader("Stemming")
        if not df_stemmed.empty and 'full_text' in df_stemmed.columns:
            st.write("Displaying pre-stemmed data from `StemmingJumbo.csv`:")
            st.dataframe(df_stemmed[['full_text']].head())
            st.info("Note: For live stemming, you would integrate a stemmer like Sastrawi's StemmerFactory here.")
        else:
            st.warning("Pre-stemmed data (`StemmingJumbo.csv`) not found or missing 'full_text' column. "
                       "To perform live stemming, you would need to implement Sastrawi's StemmerFactory.")
            st.info("Example of Sastrawi Stemmer integration (uncomment and adapt if needed):")
            st.code("""
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# def stem_text(text):
#     if not isinstance(text, str):
#         return ""
#     return stemmer.stem(text)
# df_processed['stemmed_text'] = df_processed['full_text'].apply(stem_text)
            """)

        st.subheader("Word Cloud of Processed Text")
        all_words = ' '.join([str(text) for text in df_processed['full_text']])
        if all_words:
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(all_words)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No text available to generate word cloud. Please apply preprocessing steps.")


# --- Labeling Page ---
elif menu_selection == "Labeling":
    st.title("Sentiment Labeling")
    st.write("This section shows the sentiment labels (Positive, Neutral, Negative) based on TextBlob analysis.")

    if not df_translated_labeled.empty and 'english_tweet' in df_translated_labeled.columns:
        st.subheader("Data with English Translation and Sentiment Labels")
        st.dataframe(df_translated_labeled.head())

        # Perform TextBlob analysis and get counts
        data_tweet = list(df_translated_labeled['english_tweet'])
        total_positif = 0
        total_negatif = 0
        total_netral = 0
        total = 0

        for tweet in data_tweet:
            analysis = TextBlob(str(tweet)) # Ensure tweet is string
            if analysis.sentiment.polarity > 0.0:
                total_positif += 1
            elif analysis.sentiment.polarity == 0.0:
                total_netral += 1
            else:
                total_negatif += 1
            total += 1

        st.subheader("Sentiment Distribution")
        st.write(f"**Positive:** {total_positif}")
        st.write(f"**Neutral:** {total_netral}")
        st.write(f"**Negative:** {total_negatif}")
        st.write(f"**Total Data:** {total}")

        # Plotting the distribution
        sentiment_counts = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [total_positif, total_netral, total_negatif]
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, ax=ax, palette='viridis')
        ax.set_title('Sentiment Distribution of Tweets')
        st.pyplot(fig)

    else:
        st.warning("Translated and labeled data (`translateJumboo.csv`) not found or missing 'english_tweet' column. "
                   "Please ensure this file is available and correctly formatted.")
        st.info("To perform live translation and labeling, you would need to integrate a translation API (e.g., Google Translate API) and TextBlob.")

# --- Classification Page ----------------------------------------------------
elif menu_selection == "Classification":
    st.title("Naive Bayes Classification")
    st.write("Train a Naive Bayes model on the sentiment‑labeled data.")

    # 1) Salin dan rapikan DataFrame
    df_clf = df_translated_labeled.copy()
    df_clf.columns = df_clf.columns.str.strip().str.lower()

    # Hilangkan kolom index otomatis jika ada
    if 'unnamed: 0' in df_clf.columns:
        df_clf.drop(columns=['unnamed: 0'], inplace=True)

    # 2) Pastikan nama kolom pas
    if 'english_tweet' not in df_clf.columns:
        cand = [c for c in df_clf.columns if 'tweet' in c]
        if cand:
            df_clf.rename(columns={cand[0]: 'english_tweet'}, inplace=True)
    if 'label' not in df_clf.columns:
        cand = [c for c in df_clf.columns if 'label' in c]
        if cand:
            df_clf.rename(columns={cand[0]: 'label'}, inplace=True)

    # 3) Debug ringkas
    with st.sidebar.expander("🛠 Debug Classification"):
        st.write("Columns:", df_clf.columns.tolist())
        st.write("Shape:", df_clf.shape)
        st.dataframe(df_clf.head())

    required_cols = {'english_tweet', 'label'}
    if not df_clf.empty and required_cols.issubset(df_clf.columns):

        # Buang baris kosong
        df_clf = df_clf.dropna(subset=['english_tweet', 'label'])
        if df_clf.empty:
            st.error("Semua baris kosong setelah drop NaN.")
            st.stop()

        # 💡 Normalisasi label ➜ huruf kecil & strip
        df_clf['label'] = df_clf['label'].astype(str).str.lower().str.strip()

        # Siapkan X‑y
        X = df_clf['english_tweet'].astype(str)
        y = df_clf['label']

        # TF‑IDF
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )

        st.write(f"Training size: {X_train.shape[0]}")
        st.write(f"Test size: {X_test.shape[0]}")
        st.write(f"TF‑IDF features: {X_tfidf.shape[1]}")

        # Train model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Naive Bayes model trained!")

        # Contoh prediksi
        sample_df = pd.DataFrame({
            'Text': tfidf_vectorizer.inverse_transform(X_test[:5]),
            'True': y_test.iloc[:5].values,
            'Pred': y_pred[:5]
        })
        st.subheader("Sample predictions")
        st.dataframe(sample_df)

        # Simpan ke session_state
        st.session_state.update({
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        })

    else:
        st.warning("Kolom 'english_tweet' dan/atau 'label' tidak ditemukan.")
        st.info("Periksa file di menu 'Labeling' atau lihat debug di sidebar.")

# --- Model Evaluation Page --------------------------------------------------
elif menu_selection == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Evaluate the performance of the trained Naive Bayes model.")

    if all(k in st.session_state for k in ['model', 'X_test', 'y_test', 'y_pred']):
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']

        st.subheader("Evaluation Metrics")
        st.write(f"**Accuracy:**  {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        st.write(f"**Recall:**    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        st.write(f"**F1‑Score:**  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        labels_cm = ['positif', 'netral', 'negatif']   # sesuai label lower‑case
        cm = confusion_matrix(y_test, y_pred, labels=labels_cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_cm, yticklabels=labels_cm, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    else:
        st.warning("Model not trained yet. Silakan latih di menu 'Classification' terlebih dahulu.")
