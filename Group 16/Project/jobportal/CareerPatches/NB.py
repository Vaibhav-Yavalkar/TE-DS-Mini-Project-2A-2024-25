import pickle


def NB(skills):

    with open(r'CareerPatches\naive_bayes_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open(r'CareerPatches\vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    #skills_input = ["TensorFlow, NLP, Machine Learning, Python"]  

    skills_input_vectorized = loaded_vectorizer.transform(skills)


    predicted_job_title = loaded_model.predict(skills_input_vectorized)

    print("Predicted Job Title:", predicted_job_title[0])
    return predicted_job_title[0]
