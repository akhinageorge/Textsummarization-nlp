# Textsummarization-nlp

This Project focuses on the creation of an NLP pipeline which summarises newspaper articles scarped from various sources. Any NLP Pipeline construction employs the steps - Data Acquisition, Text Preprocessing, Featured Engineering, Modelling and Model Evaluation. 

Step 1: Data Acquisition

The Data Set is employed as a CSV file uploaded in a google drive. Imported data from the url and read as CSV using Pandas.

Step 2: Text Preprocessing

This step included processing the text - cleaning included conversion to smaller case, remove all special characters, punctuation.

Step 3: Featured Engineering

This step included the conversion of text data to numerical data, The method employed is TF-IDF algorithm. The sentences were tokenised into words and a frequency distribution made. Sentence scores were created which is later used in creating the summary. All of these were done as functions which could be later employed in modelling.

Step 4: Modelling

One paragraph from the specified row (indexed) is assigned to a variable "para". num_sent is preset to 5, this is a variable which can be varied depending on the number of sentences to beconsidered.. Text processing and TF-IDF calculations are made by passing the variable to the functions and summary is generated.

Step 5: Model Evaluation 

The Original content and the no. of words in it is printed out initially. The summary and the no. of words in it is printed next. A comparison can be drawn as such. 
