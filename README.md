<h1>Exploring Emotional Progression in Text Using a Hybrid BERT-DSL Model</h1>
Motivation / Introduction
This project explores an advanced form of sentiment analysis that extends beyond traditional sentence-level classification. Real-world textual data — such as customer reviews, social media posts, and feedback — often exhibits varying emotions across different parts of a paragraph.

We introduce a novel modeling layer called the Delta Sentiment Layer (DSL). When used in conjunction with a BERT-based architecture, it tracks emotional progression within a paragraph, offering a richer and more nuanced interpretation of sentiment.

Scope of the Project
The primary goal is to develop a paragraph-level sentiment analysis model capable of identifying and visualizing sentiment transitions within text. Unlike traditional models that classify sentiment only at a sentence or document level, this model segments text into equal parts and applies a custom emotional shift tracking mechanism.

<h2>Applications include:</h2>        

-Marketing insights

-Political discourse analysis

-Content review systems

<h2>Methodology</h2> 

<h3>1. Dataset Preparation</h3>        
Dataset: Sentiment140 (1.6M tweets, labeled positive or negative)

Preprocessing: remove URLs, mentions, non-English text; lowercase conversion; expand contractions

<h3>2. Model Selection</h3>        
Compared logistic regression, Naïve Bayes, SVM, and BERT

Found BERT most effective for building the hybrid model

</h3>3. Paragraph Segmentation</h3>        
Paragraph (or simulated paragraph from tweets) split into three equal segments

Padding for uniform input length

<h3>4. Model Architecture</h3>        
BERT-base-uncased generates contextual embeddings

Segments: start, middle, end → mean embeddings computed for each

Delta computation: difference between end and start embeddings
<img width="692" height="461" alt="image" src="https://github.com/user-attachments/assets/4035533b-0d25-4d70-b35c-a69978a213e0" />


<h3>5. Sentiment Visualization</h3>        
Mean embeddings for each segment classified separately
Sigmoid scores plotted to show sentiment trajectory (start → middle → end)

Hybrid BERT-DSL outperforms baseline BERT
<img width="823" height="422" alt="image" src="https://github.com/user-attachments/assets/496bfbc5-e089-4692-bbdb-ee87780636f5" />

<img width="687" height="578" alt="image" src="https://github.com/user-attachments/assets/7df0458d-637d-4b06-b099-46f415bbd0a1" />



Achieved accuracy: 89.95% and F1-score: 0.90

Visualization reveals emotional shifts like anger → resolution or excitement → neutrality
Conclusion
The hybrid BERT-DSL model successfully tracks emotional shifts at a paragraph level, providing a deeper, more human-like understanding of sentiment. 

<h2>Future work includes:</h2>        

Multilingual support

Real-time application integration

Refinement for longer documents and complex discourse

