import pickle

with open('naive_bayes_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

skills_input = ["Virtualization, Windows Server, Networking, Cybersecurity, Database Management, Troubleshooting"]  

skills_input_vectorized = loaded_vectorizer.transform(skills_input)


predicted_job_title = loaded_model.predict(skills_input_vectorized)

print("Predicted Job Title:", predicted_job_title[0])
