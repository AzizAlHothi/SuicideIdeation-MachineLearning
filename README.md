# Machine Learning Model to explore Suicide Ideation and Depression posts online
A project featuring a pipeline of supervised and unsupervised machine learning model aimed at understanding the language that delineates suicide ideation posts from general depression posts online.
### Quick Background
This is Abdulaziz AlHothi's project of building a machine learning model that can improve suicide detection through lingual indications on social media. The [dataset](https://https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) (specifically Cleaned Suicidewatch Vs. Depression) imported into this project draws texts from SuicideWatch reports and depression forums , classing them into two different classes "SuicideWatch" and "Depression".


## Note 0: Improving dataset selection
Earlier feedback to this project was that it looks at a dataset that compares text from r/suicide, to texts from r/depression and r/teenagers. 

However, to improve the scope of this project, I will be using an alternative e dataset (from the same source) which compares posts marked for suicidewatch (directly looking at very high risk of suicide ideation) and posts from r/depression. The goal is to create a machine learning model that can differentiate between generic posts on r/depression and suicidewatch listed texts from r/suicide. 

The overarching goal is of course for this project to act as a first step in building a more versatile machine learning model that is able to dileate text that is highly indicative of suicide ideation from text that is experessing depression.  

#### Limitations 

Separating suicidewatch from depression is, at a conceptual level, difficult. Theoretically, depression is a primary cause of suicide ([World Health Organization, 2014](https://www.who.int/publications/i/item/9789241564779)), so any text that expresses depression, is theoretically indicative of suicide. 

In practical terms, A lot of overlap between the two classes is to be expected. The goal is of course to achieve as much accurate separation as possible. 

Another limitation lies in the internal/construct validity of the research proposal. While we are aiming to dileneate suicide-ideation text from depression-text, we have to address that the content of the r/depression is not strictly all expressive of depression. A lot of posts and comments will express support, recovery and improvement in mood. While this is a limitation that is difficult to overcome within this project a lone, the unsupervised portion of this project will serve as an exploratory assessment to uncover what are the salient topics in both classes, and identify which topics are overlap, which topics describe their respective classes accurately, and which topics can be good separators of the classes. 


# Supervised Learning
The supervised learning part of this project will compare a tree-based model (random forest classifier) and a linear-based model (logistic regression). Using a GridSearchCV to optimize hyperparameter finetuning, I will fairly contrast the two classifiers at their best. 

#### Note 3.3: Discussion of RF Classifier 

Through optimizing the hyperparameters for the RandomForest Classifier , we were able to achieve about 77% accuracy and recall. Not terrible numbers , but not super great. Keep in mind that we are more concerned about Type I error (missing potential suicide-ideation texts). 

The top 20 features were also revealing. Odd features that stood out were the numerical features (avg_word_len_cap) and (char_count_cap). We'll see if this becomes a recurring theme throughout the logistic regression. 

#### Note 3.4: A Look at Logistic Regression Classifier
Logistic Regression classifier has the upperhand not only in generic accuracy methods, but in avoiding the errors that are most harmful for this purpose of this project (Type I: False negatives).


# Unsupervised Learning

In the second part of this project, I will employ unsupervised learning models (K-means and LDA) to reveal what topics and subtexts are featured the most in the sample of reddit posts. 

## K-Means 
### Determining the number of clusters

by graphing the Mean Squared Error (MSE) for the number of clusters, we will be able to find the optimal number of clusters by finding the number of clusters which most significantly reduces the MSE. This is also known as the Elbow Method. (Based on the image below , 5 clusters were chosen). 
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/fb6ad3fb-7cca-47fc-9eac-e36847869dff" />

### Evaluating the K-means results 
Using the following evaluation methods, we determined the efficacy (or lack thereof) of the k-means in this specific application: 
NMI Score:          0.0071
Homogeneity Score:  0.0109
Completeness Score: 0.0053
V-measure Score:    0.0071

### K-means Verdict
The metric results are subpar (very close to zero). Furthermore, the topics identified did not help us separate the two topics (suicidewatch and depression). This is normal considering the linguistic closeness we observed earlier in this pipeline

## LDA
The pyLDAvis package was used on this sample. The coherence score was calculated to determine the optimal number of topics (K) (12 topics were selected)

<img width="696" height="493" alt="image" src="https://github.com/user-attachments/assets/e69a7dbd-8bab-4d0a-8176-c9ae7caad34d" />

using the pyLDAvis package , a map with two principal components is visualized to illustrate the distance between the clusters of topics across the two axes (lambda is set 1).

<img width="1179" height="745" alt="image" src="https://github.com/user-attachments/assets/fb71aaff-4225-4a47-8eb3-d55971aa7f70" />

## LDA Vs. K-means (in this application)

--- Comparing K-Means and LDA Clustering Metrics (vs Ground Truth) ---

K-Means Metrics:
  NMI Score:          0.0071
  Homogeneity Score:  0.0109
  Completeness Score: 0.0053
  V-measure Score:    0.0071

LDA Metrics (based on dominant topic):
  NMI Score:          0.0528
  Homogeneity Score:  0.1103
  Completeness Score: 0.0347
  V-measure Score:    0.0528

## Unsupervised model verdict
### Quantitatively Speaking 
The Adjust Rand score BARELY outscores the k - means method , but still better. However, I ran the entire evaluation to compare both k-means and LDA model from evaluation metrics presented in class , and LDA beats k-Means on every metric. 

### Qualitatively Speaking
Since the topics in the LDA model were much more coherent and made sense for exploration, I will use the LDA topics over the k-Means. I will run a code chunk below to measure the dominant topic in each document (reddit post) and then look at the distribution of classes across topics and determine which topics are salient in suicide labels. This will allow for a more qualitative approach to identifying and dileneating suicide ideation in text. 

LDA wins this part of the pipeline.


<img width="1175" height="786" alt="image" src="https://github.com/user-attachments/assets/28ee4994-a9bc-4c01-87bd-564bc1582ef9" />

--- Topic Legend ---
Topic 1: Suicidal Thoughts and Urge to Die
Topic 2: Family Conflict and Parental Struggles
Topic 3: Clinical Depression and Medication Management
Topic 4: Philosophical Reflections and Meaning of Life
Topic 5: Seeking Help and Suicidal Ideation
Topic 6: Timeline of Struggles and Personal History
Topic 7: Loneliness and Interpersonal Relationships
Topic 8: Body Image and Perceptual Pain
Topic 9: Explosive Anger and Despair
Topic 10: Hopelessness and Desire to Escape Life
Topic 11: Emotional Turmoil and Confusion
Topic 12: Academic and Work-Related Stress

# Results and Discussion

With this new dataset, it seems that the separation is becoming harder and harder to achieve with two very close topics in concept (depression and suicide). Let's first review the unsupervised learning section to uncover which topics seem to divide the two topics 

## Unsupervised Learning Dileneating Topics

1. Topic 2: Family Trauma and Parental Struggles (~63% SuicideWatch)

This topic achieves above 50% of being relevant in dileneating suicidewatch posts from depression, which could point to the depth of a family trauma and isolation dominant topic in a text. 

2. Topic 1: Suicidal Thoughts and Urge to Die (85% SuicideWatch)

Not a surprising outcome , proves as a good sanity check for my model. 

3. Topic 10: Hopelessness and Desire to Escape Life (73% SuicideWatch)

Not a surprising outcome , although people with MDD can also express this emotion. This seems to be very latent in suicidewatch-listed posts. 

4. Topic 9: Explosive Anger and Despair (73% SuicideWatch)

It seems that anger and outrage , combined with suicidal thoughts, seem to give the sense of urgency and seriousness that the person is considering to end their life. 

5. Topic 7: Lineliness and interpesonal relations (75% Depression)

Also not surprising, as people with depression will express isolation and mental health struggles with their family more pronouncely than posts listed on suicidewatch. 

6. Topic 3: Clinical Depression and Medication Management (71% Depression)

A good sanity check for our model



# Conclusion
the unsupervised topic modeling added valuable depth to my understanding of the latent structures within the data. Specifically, it helped reveal nuanced patterns that might not be immediately apparent from class labels alone—such as the unexpected clustering of existential reflection and academic stress more heavily within depression posts, and the distinct presence of rage and family trauma in suicide-related texts.

This insight suggests that unsupervised learning can serve as a powerful complementary tool—not just for exploratory analysis but also for improving classification pipelines. By incorporating topic proportions or topic-informed features into my supervised models, I may be able to better capture semantic context, ultimately enhancing both overall accuracy and, more importantly, recall. This matters critically in this domain, as improving recall helps reduce the risk of false negatives (Type II errors), which is essential in a mental health detection setting where missing high-risk posts could have serious consequences.

In future iterations, I plan to explore hybrid models that combine topic-based embeddings with classification, allowing for more context-aware and interpretable predictions.
