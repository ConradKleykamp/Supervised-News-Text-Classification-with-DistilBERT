# Supervised-News-Text-Classification-with-DistilBERT
Classifying news articles about 'health' or 'wellness' using DistilBERT

![image](https://github.com/user-attachments/assets/afdef240-2831-43e8-9fb2-36ce5c480e4d)

---

### Objective
This project was completed as part of CU Boulder's Supervised Text Classification for Marketing Analytics final project. Overall, the project emulates a fictional contextual advertising problem, i.e. an arbitrary company wishes to identify as many news articles that mention 'health' or 'wellness' for a new media campaign. By successfully identifying relevant news articles, this company could then identify relevant URLs and then deploy their media campaign to those sites. In order to accomplish this, this project leverages ktrain and a DistilBERT model for supervised text classification. BERT (Bidirectional Encoder Representations from Transformers) is a language representation model which can be used for natural language processing tasks. BERT can often be computationally expensive and thus DistilBERT will be used instead. DistilBERT retains BERT's exceptional language understanding capabilities but trades a small amount of performance for a reduction in size and increase in speed. Overall, the model will attempt to classify whether or not a given news article (based upon its headline and short description text) falls into a 'health'/'wellness' category. The model's performance will be measured by its precision, recall, and f1-score on a validation set. These metrics will be compared to a benchmark model, created by the professor for this course. Lastly, the model will be leveraged to make predictions on novel, fictional headlines.

The dataset used for this project is available on Kaggle and was uploaded by Rishabh Misra. The dataset contains roughly 210,000 news headlines dating from 2012 to 2022. All news articles are from HuffPost. Each news article has the following attributes: category, headline, authors, link, short_description, date. The dataset is publicly available under the CC BY 4.0 license. 

This project was originally completed in Google Colab using the Tesla T4 GPU. In order to reduce the runtime of this notebook, I have opted to comment out the lines of code in which the standard GPU takes way too long to run. Moreover, I have also included images from my original work to show the results of these commented out lines of code. If you would like to view the full, original work, please feel free to check out my project on Google Colab (https://colab.research.google.com/drive/1beGNV43zvYgbllXk_Y7lBx-BPVi5EPqH?usp=sharing). Also, if you would like to fully emulate this notebook, I would recommend using an accelerator similar to Google Colab's Tesla T4 GPU.

---

### Methods
Libraries Used
- os
- pandas
- numpy
- ktrain (text, texts_from_df)
- tf_keras

Data Preprocessing
- Creating a new column ('combined_text') which combines the 'headline' and 'short_description' columns
- Creating a new column ('healthy') which yields a 1 (positive case) for all news articles that have the category 'WELLNESS' or 'HEALTHY LIVING'
- All news articles that do not have one of these categories have a 0 in the 'healthy' column (negative case)
- Balancing the positive and negative cases by selecting a relevant sample size

Building the Model
- Target names: 'NOT HEALTHY', 'HEALTHY'
- Model parameters:
    - max_features = 20000
    - maxlen = 128
    - val_pct = 0.1
    - ngram_range = 1
    - preprocess_model = 'distilbert'
- Using learner.lr_find() to optimize the learning rate
    - batch_size = 8
    - max_epochs = 6
- Applying optimal learning rate to model (1e-4)

Model Validation
- Precision, recall, f1-score, support

Predictions
- Creating novel headlines to make predictions on

---

### General Results
Model Performance: Comparison to Benchmark (Benchmark scores shown in italics)
Positive Class:
- Precision --> 0.92 | *0.85*
- Recall --> 0.93 | *0.89*
- f1-score --> 0.93 | *0.87*

Negative Class:
- Precision --> 0.93 | *0.88*
- Recall --> 0.91 | *0.84*
- f1-score --> 0.92 | *0.86*

Accuracy:
- 0.92 | *0.86*

In conclusion, the model created in this project outperformed the benchmark model across all metrics. For both classes, all metrics of the model exceeded 90%, suggesting that the final model is effective at correctly classifying the news articles into the appropriate classes. For real-world applications, these metrics suggest that the model may be sufficient. However, it is likely that other models may be equally effective or even superior for this supervised text classification task. One such model could be BERT. Although BERT is more resource intensive than the slimmer DistilBERT, it is likely that it would yield better results.
