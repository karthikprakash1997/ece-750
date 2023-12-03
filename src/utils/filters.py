from openai import OpenAI
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

openai_api_key = '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'
organization_id = 'org-23QOmoGAasgqLil6LpPNL9Ev' # Replace with your actual API key
client = OpenAI(api_key=openai_api_key)

model_filename = 'random_forest_model.joblib'
# with open(model_filename, 'rb') as f_in:
# trained_model = joblib.load(r'random_forest_model.joblib')
trained_model = joblib.load(r'C:\Users\91900\Desktop\karthik_prakash\UWaterloo\ECE_750\Project\final-projct-back-end\src\utils\random_forest_model.joblib')
# print(foob)  
# trained_model = joblib.load(model_filename)

# try:
# #    est = pickle.loads("model_from_prevision_version.pickle")
#    trained_model = joblib.load(r'C:\Users\91900\Desktop\karthik_prakash\UWaterloo\ECE_750\Project\final-projct-back-end\src\utils\random_forest_model.joblib')
# except InconsistentVersionWarning as w:
#    print(w.original_sklearn_version)

data = {
    'Mental health diagnosis one': [
        # 'Missing/unknown/not collected/invalid/no or deferred diagnosis',
        'Trauma- and stressor-related disorders',
        'Personality disorders',
        'Schizophrenia or other psychotic disorders',
        #'Alcohol or substance use disorders',
        # 'Other disorders/conditions',
        # 'Anxiety disorders',
        # 'Attention deficit/hyperactivity disorder (ADHD)',
        # 'Conduct disorders',
        # 'Delirium, dementia',
        # 'Bipolar disorders',
        # 'Depressive disorders',
        # 'Oppositional defiant disorders',
         'Pervasive developmental disorders'
    ],
    'Reporting state code': [
        'New England', 'Middle Atlantic', 'East North Central', 'West North Central',
        #'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific'
    ],
    '0-11 years': [20, 794, 7891, 19156],
    '12-14 years': [20, 794, 7891, 19156],
    '15-17 years': [20, 794, 7891, 19156],
    '18-20 years': [20, 794, 7891, 19156],
    '21-24 years': [20, 794, 7891, 19156],
    '25-29 years': [20, 794, 7891, 19156],
    '30-34 years': [20, 794, 7891, 19156],
    '35-39 years': [20, 794, 7891, 19156],
    '40-44 years': [20, 794, 7891, 19156],
    '45-49 years': [20, 794, 7891, 19156],
    '50-54 years': [20, 794, 7891, 19156],
    '55-59 years': [20, 794, 7891, 19156],
    '60-64 years': [20, 794, 7891, 19156],
    '65 years and older': [20, 794, 7891, 19156],
    'Full-time': [20, 794, 7891, 19156],
    'Part-time': [20, 794, 7891, 19156],
    'Employed full-time/part-time not differentiated': [20, 794, 7891, 19156],
    'Unemployed': [20, 794, 7891, 19156],
    'Not in labor force': [20, 794, 7891, 19156],
    'Retired, disabled': [20, 794, 7891, 19156],
    'Student': [20, 794, 7891, 19156],
    'Homemaker': [20, 794, 7891, 19156],
    'Sheltered/non-competitive employment': [20, 794, 7891, 19156],
    'Other': [20, 794, 7891, 19156],
    'Special education': [20, 794, 7891, 19156],
    '0 to 8': [20, 794, 7891, 19156],
    '9 to 11': [20, 794, 7891, 19156],
    '12 (or GED)': [20, 794, 7891, 19156],
    'More than 12': [20, 794, 7891, 19156],
    'Never married': [20, 794, 7891, 19156],
    'Now married': [20, 794, 7891, 19156],
    'Separated': [20, 794, 7891, 19156],
    'Divorced, widowed': [20, 794, 7891, 19156],
    'American Indian/Alaska Native': [20, 794, 7891, 19156],
    'Asian': [20, 794, 7891, 19156],
    'Black or African American': [20, 794, 7891, 19156],
    'Native Hawaiian or Other Pacific Islander': [20, 794, 7891, 19156],
    'White': [20, 794, 7891, 19156],
    'Some other race alone/two or more races': [20, 794, 7891, 19156],
    'Homeless': [20, 794, 7891, 19156],
    'Private residence': [20, 794, 7891, 19156],
    'Other': [20, 794, 7891, 19156],
    'Male': [20, 794, 7891, 19156],
    'Female': [20, 794, 7891, 19156],
    'Yes': [20, 794, 7891, 19156],
    'No': [20, 794, 7891, 19156],
    'Other jurisdictions': [20, 794, 7891, 19156],
    'Intrusive Memories': [ 1, 0, 0, 0],
    'Negative Changes in Mood/Thinking': [1, 0, 0, 0],
    'Changes in Emotional Reactions': [ 1, 0, 0, 0],
    'Impulsivity': [1, 0, 0, 0],
    'Shame': [0, 1, 0, 0],
    'Anger': [0, 1, 0, 0],
    'Hallucinations': [0, 1, 0, 0],
    'Delusions': [0, 1, 0, 0],
    'Disorganized Thinking': [0, 0, 0, 0],
    'Reduced Emotional Expression': [0, 1, 0, 0],
    'Cravings': [0, 0, 0, 0],
    'Loss of Control': [ 1, 0, 0, 0],
    'Physical Dependence': [0, 1, 0, 0],
    'Withdrawal Symptoms': [0, 1, 0, 0],
    'Excessive Worry': [0, 1, 0, 0],
    'Restlessness': [0, 0, 0, 0]
}

df = pd.DataFrame(data)



def generate_openai_explanation(api_response_openai):
    response = client.chat.completions.create(
        engine="text-davinci-002",
        prompt=f"Explain the diagnosis and provide recommendations based on the following information:\n{api_response_openai}\n",
        max_tokens=150
    )
    explanation_text = response['choices'][0]['text'].strip()

    return explanation_text

def generate_chatgpt_response(api_response_openai):

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a mental health assistant, providing additional insights based on user input."},
            {"role": "user", "content": f"Provide insights based on the following information:\n{api_response_openai}\n"}
        ]
    )

    chatgpt_response = completion.choices[0].message.content.strip()

    return chatgpt_response

def identify_diagnosis(model, user_data, label_encoder_mapping):

    user_df = pd.DataFrame([user_data])
    label_columns = ['Reporting state code']

    for column in label_columns:
        user_df[column] = label_encoder_mapping[column].transform(user_df[column])

    user_features = user_df
    diagnosis_code = model.predict(user_features)[0]
    predicted_diagnosis = label_encoder_mapping['Mental health diagnosis one'].inverse_transform([diagnosis_code])[0]

    return predicted_diagnosis

def generate_api_response_openai(predicted_diagnosis, recommendations=None):
    prompt = f"Your estimated diagnosis, based on the input you provided, is {predicted_diagnosis}."

    if recommendations:
        prompt += f" There are some recommendations that you can use for improvement of your state: {recommendations}. Based on your place and location, you can reach such sources (provide text here)."
        prompt += f" Keep in mind these recommendations: {recommendations}"

    prompt += f"\n\nProvide an explanation for the identified diagnosis and recommendations using OpenAI API:\n"
    return prompt

def suggestions():
        example_input = {'Reporting state code': 'West North Central', '0-11 years': '20', '12-14 years': '794', '15-17 years': '7891', '18-20 years': '19156', '21-24 years': '0', '25-29 years': '1508', '30-34 years': '3462', '35-39 years': '90', '40-44 years': '200', '45-49 years': '100398', '50-54 years': '69', '55-59 years': '3640', '60-64 years': '1169', '65 years and older': '10187', 'Full-time': '7690', 'Part-time': '7370', 'Employed full-time/part-time not differentiated': '15849', 'Unemployed': '279', 'Not in labor force': '2965', 'Retired, disabled': '519', 'Student': '0', 'Homemaker': '1', 'Sheltered/non-competitive employment': '1', 'Other': '1', 'Special education': '1', '0 to 8': '1', '9 to 11': '0', '12 (or GED)': '1', 'More than 12': '0', 'Never married': '1', 'Now married': '0', 'Separated': '1', 'Divorced, widowed': '0', 'American Indian/Alaska Native': '1', 'Asian': '0', 'Black or African American': '1', 'Native Hawaiian or Other Pacific Islander': '0', 'White': '1', 'Some other race alone/two or more races': '0', 'Homeless': '1', 'Private residence': '0', 'Other': '1', 'Male': '0', 'Female': '1', 'Yes': '0', 'No': '1', 'Other jurisdictions': '1', 'Intrusive Memories': '0', 'Negative Changes in Mood/Thinking': '1', 'Changes in Emotional Reactions': '0', 'Impulsivity': '1', 'Shame': '0', 'Anger': '1', 'Hallucinations': '0', 'Delusions': '1', 'Disorganized Thinking': '0', 'Reduced Emotional Expression': '1', 'Cravings': '0', 'Loss of Control': '1', 'Physical Dependence': '0', 'Withdrawal Symptoms': '1', 'Excessive Worry': '0', 'Restlessness': '1'}
        label_encoder_mapping = {}
        for column in ['Mental health diagnosis one', 'Reporting state code']:
            label_encoder_mapping[column] = LabelEncoder()
            label_encoder_mapping[column].fit(df[column])
        
        predicted_diagnosis = identify_diagnosis(trained_model, example_input, label_encoder_mapping)
        api_response_openai = generate_api_response_openai(predicted_diagnosis, None)
        print(api_response_openai, predicted_diagnosis)

        try:
            chatgpt_response = generate_chatgpt_response(api_response_openai)

            # print("\nExample input used for testing:")
            # for question, answer in example_input.items():
                # print(f"{question}: {answer}")
            # print(f"\nAPI Response (OpenAI): {api_response_openai}\n")
            # print(f"ChatGPT Response: {chatgpt_response}\n")
         
            return chatgpt_response

        except Exception as e:
            if "429" in str(e):
                print(str(e))
            else:
                print(f"OpenAI API error occurred. Skipping the current iteration. Error: {e}")


def feedback(api_response_openai, response):
    chatgpt_response = generate_chatgpt_response(api_response_openai, response)
    return chatgpt_response
