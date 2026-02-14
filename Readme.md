# NLP Classifier

## Overview

This project implements a simple Natural Language Processing (NLP) classifier. The goal is to classify text documents into predefined categories.

Currently, the classifier distinguishes between:

- **Politics**
- **Sport**

## Project Structure

- `politics/` → Contains training documents for the politics category  
- `sport/` → Contains training documents for the sports category  
- `NLP_classifier.py` → Script that trains and runs the classifier  

## How It Works

1. The script reads text files from both folders.  
2. It preprocesses the text, such as tokenization and cleaning.  
3. Features are extracted, using methods like Bag of Words or TF-IDF.  
4. A machine learning model is trained.  
5. The model predicts the category of new input text.  

## How to Run

Navigate to the project directory and run:

```bash
python NLP_classifier.py
```