# DevelopersHub AI/ML Engineering – Advanced Internship Tasks (Aug 2025)

This repository contains the complete set of advanced AI/ML tasks completed as part of the **DevelopersHub Corporation AI/ML Engineering Internship**. The internship focused on hands-on experience with state-of-the-art machine learning and artificial intelligence techniques, including **transformers, ML pipelines, multimodal learning, conversational AI, and LLM applications**.

All tasks are implemented in **Google Colab**, with free-to-use libraries and models, ensuring reproducibility and accessibility.  

---

## **Table of Contents**

1. [Task 1: News Topic Classifier Using BERT](#task-1-news-topic-classifier-using-bert)  
2. [Task 2: End-to-End ML Pipeline for Customer Churn](#task-2-end-to-end-ml-pipeline-for-customer-churn)  
3. [Task 3: Multimodal ML – Housing Price Prediction](#task-3-multimodal-ml--housing-price-prediction)  
4. [Task 4: Context-Aware Chatbot Using LangChain/RAG](#task-4-context-aware-chatbot-using-langchainrag)  
5. [Task 5: Auto Tagging Support Tickets Using LLM](#task-5-auto-tagging-support-tickets-using-llm)  

---

## **Task 1: News Topic Classifier Using BERT**

- **Objective:**  
  Fine-tune a transformer model (BERT) to classify news headlines into topic categories.  

- **Dataset:**  
  AG News Dataset (Hugging Face Datasets)

- **Methodology / Approach:**  
  - Tokenized and preprocessed the dataset.  
  - Fine-tuned `bert-base-uncased` using Hugging Face Transformers.  
  - Evaluated the model using **accuracy** and **F1-score**.  
  - Deployed the model with **Streamlit** for live interaction.  

- **Key Results / Observations:**  
  - Achieved **~92% accuracy** on the test set.  
  - Model can classify live input headlines into four categories: `World`, `Sports`, `Business`, `Sci/Tech`.  
  - Lightweight and deployable on Streamlit.

---

## **Task 2: End-to-End ML Pipeline for Customer Churn**

- **Objective:**  
  Build a reusable and production-ready ML pipeline to predict customer churn.  

- **Dataset:**  
  Telco Churn Dataset  

- **Methodology / Approach:**  
  - Implemented **data preprocessing** (scaling, encoding) using scikit-learn `Pipeline`.  
  - Trained **Logistic Regression** and **Random Forest** models.  
  - Hyperparameter tuning using **GridSearchCV**.  
  - Exported the complete pipeline using **joblib** for reusability.  

- **Key Results / Observations:**  
  - Random Forest achieved **87% accuracy** on test data.  
  - Pipeline is modular, reusable, and ready for production deployment.

---

## **Task 3: Multimodal ML – Housing Price Prediction**

- **Objective:**  
  Predict housing prices using both structured tabular data and house images.  

- **Dataset:**  
  Housing Sales Dataset + Custom Image Dataset  

- **Methodology / Approach:**  
  - Extracted features from images using **Convolutional Neural Networks (CNNs)**.  
  - Combined image features with tabular data (numerical and categorical).  
  - Trained a regression model using both modalities.  
  - Evaluated performance using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.  

- **Key Results / Observations:**  
  - Model achieved **MAE ~\$12,000** and **RMSE ~\$18,500** on test set.  
  - Demonstrated effectiveness of **multimodal feature fusion** for prediction.

---

## **Task 4: Context-Aware Chatbot Using LangChain/RAG**

- **Objective:**  
  Build a conversational chatbot that can **remember context** and retrieve answers from a knowledge base.  

- **Dataset:**  
  Custom text corpus (internal documentation & knowledge base)

- **Methodology / Approach:**  
  - Used **LangChain** for managing conversation and context memory.  
  - Implemented **Retrieval-Augmented Generation (RAG)** using **FAISS vector store**.  
  - Integrated **free Hugging Face LLM (`google/flan-t5-small`)** for answer generation.  
  - Deployed chatbot using **Streamlit**.  

- **Key Results / Observations:**  
  - Chatbot provides **context-aware responses** even in multi-turn conversations.  
  - Successfully retrieves relevant information from custom knowledge base.  
  - Fully free-to-use, runs efficiently in Google Colab.

---

## **Task 5: Auto Tagging Support Tickets Using LLM**

- **Objective:**  
  Automatically tag support tickets into categories using a large language model.  

- **Dataset:**  
  Free-text Support Ticket Dataset  

- **Methodology / Approach:**  
  - Implemented **zero-shot classification** using Hugging Face LLM (`google/flan-t5-small`).  
  - Enhanced accuracy using **few-shot examples** in prompts.  
  - Extracted **top 3 most probable tags** per ticket.  
  - Fully automated pipeline in Google Colab.  

- **Key Results / Observations:**  
  - Correctly tagged tickets for technical issues, login problems, billing, and account management.  
  - Demonstrated ability of **LLMs for multi-class classification** without fine-tuning.  

---

## **Technologies & Tools Used**

- **Libraries:** Transformers, Torch, Scikit-learn, Pandas, Sentence-Transformers, FAISS, LangChain, Streamlit  
- **Models:** BERT (bert-base-uncased), Flan-T5 (google/flan-t5-small), CNNs  
- **Platforms:** Google Colab, Streamlit  

---

## **Usage Instructions**

1. Open any task notebook in Google Colab.  
2. Install dependencies using `!pip install` commands provided in notebooks.  
3. Run all cells sequentially for dataset loading, preprocessing, model training, and evaluation.  
4. For Streamlit apps, use:

```bash
!streamlit run app.py
