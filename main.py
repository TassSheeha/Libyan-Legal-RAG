import os
import re
import pdfplumber
from dotenv import load_dotenv

# مكتبات معالجة اللغة العربية
from arabic_reshaper import reshape
from bidi.algorithm import get_display

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- 1. تحميل الإعدادات ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 2. دالة تصحيح النصوص العربية ---
def process_arabic_text(text):
    if not text: return ""
    text = re.sub(r'[^\w\s\.\u0600-\u06FF،؛-]', ' ', text)
    reshaped = reshape(text)
    bidi_text = get_display(reshaped)
    return re.sub(r'\s+', ' ', bidi_text).strip()

# --- 3. استخراج النصوص ---
def extract_legal_documents(data_path="./data"):
    all_docs = []
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"تم إنشاء مجلد {data_path}، يرجى وضع ملفات الـ PDF داخله.")
        return []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            print(f"جاري معالجة الملف: {file}")
            try:
                with pdfplumber.open(os.path.join(data_path, file)) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        raw_text = page.extract_text()
                        if raw_text:
                            cleaned = process_arabic_text(raw_text)
                            metadata = {"source": file, "page": page_num + 1}
                            all_docs.append(Document(page_content=cleaned, metadata=metadata))
            except Exception as e:
                print(f"خطأ في قراءة الملف {file}: {e}")
    return all_docs

# --- 4. إنشاء الـ VectorStore ---
def setup_retriever(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 5. بناء الـ RAG Chain بالموديل الجديد ---
def create_legal_rag(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )

    template = """أنت مستشار قانوني خبير في التشريعات الليبية. 
أجب بناءً على السياق المرفق فقط. اذكر رقم المادة إن وجد.

السياق:
{context}

السؤال: {question}
الإجابة:"""

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 6. التشغيل ---
def main():
    if not GROQ_API_KEY:
        print("خطأ: يرجى إضافة GROQ_API_KEY في ملف .env")
        return

    docs = extract_legal_documents()
    if not docs:
        print("لا توجد مستندات للمعالجة.")
        return

    print("بناء قاعدة البيانات...")
    retriever = setup_retriever(docs)
    rag_chain = create_legal_rag(retriever)

    test_questions = [
        "ما هي المسؤوليات الرئيسية للوزارات والجهات الحكومية في ليبيا فيما يخص رعاية الأطفال؟",
        "ما هي المعاهدات الدولية المتعلقة بحقوق الطفل التي صادقت عليها ليبيا؟",
        "كم قيمة رأس مال المصرف وفقاً للقانون رقم 46؟",
        "ما هي المؤهلات التعليمية المشروطة للمدير العام ومجلس الإدارة؟"
    ]

    for q in test_questions:
        print(f"\nسؤال: {q}")
        print(f"الرد: {rag_chain.invoke(q)}")
        print("-" * 50)

if __name__ == "__main__":
    main()