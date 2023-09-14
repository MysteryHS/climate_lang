# Readme for climate lang project

This repo contains all the code for the climate lang project. The goal of this project is to have an easy access to all information provided by the IPCC reports. 

It uses langchain to load the IPCC reports and GPT-4 to answer questions on the reports.

When a user asks a question, the most relevant parts of the text will be retrieved and used as a context for GPT-4 to answer the question.

Check the ```pdf_chat.ipynb``` for the code and reproduce the results.