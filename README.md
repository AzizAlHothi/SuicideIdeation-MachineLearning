# SuicideIdeation-MachineLearning
A project featuring a pipeline of supervised and unsupervised machine learning model aimed at understanding the language that delineates suicide ideation posts from general depression posts online.
# Suicide Detection Project
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
