"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
from nltk.corpus import stopwords

# Visuals
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
#Other 
import os

# Vectorizer
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
base_path = "https://raw.githubusercontent.com/ProfHercules/Classification_AE5_Data/main"
raw = pd.read_csv(f'{base_path}/train.csv')

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Intro Page", "EDA", "Model Page 1", "Model Page 2", "Model Page 3"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Intro Page":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	if selection =="EDA":
		st.info("This is an EDA page")
		st.markdown("This would be the explanation of the data set we used and the insights that we found :) ")
		st.header("Data Decription")
		st.text("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to\nChris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change\ncollected between Apr 27, 2015 and Feb 21, 2018. In total, 43'943 tweets were collected. Each tweet\nis labelled as one of the following classes:")
		st.subheader("Class Description")
		st.text("2 News: the tweet links to factual news about climate change\n1 Pro: the tweet supports the belief of man-made climate change\n0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change\n-1 Anti: the tweet does not believe in man-made climate change")

		#first plot
		class_map = { -1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News' }
		raw['sentiment_label'] = raw.sentiment.map(class_map)

		raw["msg_len"] = raw['message'].apply(lambda msg: len(msg))

		value_counts = (raw['sentiment'].value_counts(normalize=True) * 100.0).round(1)

		data = [
				go.Bar(
					x = raw['sentiment_label'].unique(),
					y = value_counts,
					marker={
						"colorscale":'Sunset',
						"color": value_counts
					},
					text='Percentage of class'
				)
			]

		layout = go.Layout(title='Class distribution')

		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)
		st.text("As we can see, the dataset contains about 52.3"+"% "+"of tweets that support the belief of \nman-made climate change, while there's just 17.6"+"% "+"of climate change deniers")
		# second plot -- Word Cloud
		anti = raw[raw.sentiment == -1]["message"].values

		plt.figure(figsize=(16, 13))
		wc = WordCloud(
			background_color="black",
			max_words=250, 
			stopwords=stopwords.words('english'), 
			max_font_size= 40,
			colormap='Pastel2',
			random_state=27,
		)
		wc.generate(" ".join(anti))

		plt.title("Climate Change Deniers", fontsize=20)

		plt.imshow(wc, alpha=0.98, interpolation='bilinear')
		plt.axis('off')
		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.pyplot()

			# Building out the predication page
		if selection == "Prediction":
			st.info("Prediction with ML Models")
		# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
