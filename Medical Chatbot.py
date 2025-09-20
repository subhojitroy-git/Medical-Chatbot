import streamlit as st
import pandas as pd
import requests
import json
import time

# --- Single place for API Key ---
# PASTE YOUR GOOGLE API KEY HERE. This key will be used for all AI features.
API_KEY = "AIzaSyAH4ZnzOHNl2nkjGE4WgYFHzu_T9TKKYfA"
# --- Functions for Symptom Checker ---

@st.cache_data
def load_data():
    """Loads the symptom dataset."""
    try:
        df = pd.read_csv('disease_symptoms.csv')
        df.columns = [col.lower() for col in df.columns]
        return df
    except FileNotFoundError:
        st.error("Error: 'disease_symptoms.csv' not found.")
        return None

def find_disease(user_symptoms, df):
    """Finds the most likely disease based on a list of symptoms."""
    disease_scores = {disease: 0 for disease in df['diseases']}
    for symptom in user_symptoms:
        if symptom in df.columns:
            for _, row in df.iterrows():
                if row[symptom] == 1:
                    disease_scores[row['diseases']] += 1
    max_score = max(disease_scores.values()) if disease_scores else 0
    if max_score == 0:
        return ["No specific condition matched"], user_symptoms
    predicted_diseases = [d for d, s in disease_scores.items() if s == max_score]
    return predicted_diseases, user_symptoms

def extract_symptoms_from_text(user_input, all_symptoms_list, api_key):
    """AI Step 1: Parses free text to extract a structured list of symptoms."""
    system_prompt = (
        "You are an expert medical symptom recognition AI. Analyze the user's text to identify symptoms. "
        "You will be given a list of valid symptoms. You must only return symptoms from this list. "
        "Your response MUST be a valid JSON object with a single key 'symptoms' holding a list of strings. "
        "If no valid symptoms are found, return an empty list."
    )
    full_prompt = (
        f"Valid symptoms list: {', '.join(all_symptoms_list)}\n\n"
        f"User's text: \"{user_input}\"\n\n"
        "Identify the symptoms and return them in the specified JSON format."
    )
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": { "responseMimeType": "application/json" }
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        json_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        return json.loads(json_text).get("symptoms", [])
    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        st.error("Error communicating with the symptom analysis AI.")
        print(f"DEBUG (Symptom Extraction): {e}")
        return []

def generate_symptom_summary(recognized_symptoms, predicted_diseases, api_key):
    """AI Step 2: Takes the structured results and generates a conversational summary."""
    system_prompt = (
        "You are a friendly and helpful medical information assistant. Your task is to explain the results of a symptom checker to a user "
        "in a clear, conversational, and reassuring tone. \n"
        "Follow these rules:\n"
        "1. Start by acknowledging the symptoms that were recognized.\n"
        "2. State the potential condition(s) found in the dataset that match these symptoms.\n"
        "3. **Crucially**, you MUST end with a strong, clear disclaimer that this is not a medical diagnosis and the user must consult a healthcare professional."
    )
    symptoms_str = ", ".join(recognized_symptoms)
    diseases_str = ", ".join(predicted_diseases)
    if "No specific condition matched" in diseases_str:
         full_prompt = ( f"The user's symptoms were: '{symptoms_str}'. Our system did not find a strong match. Explain this gently and advise them to see a doctor.")
    else:
        full_prompt = ( f"The user's symptoms were: '{symptoms_str}'. The potential condition(s) are: '{diseases_str}'. Explain these results conversationally.")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": full_prompt}]}],"systemInstruction": {"parts": [{"text": system_prompt}]},}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        st.error("Error generating the summary.")
        print(f"DEBUG (Summary Generation): {e}")
        return "There was an error generating the summary. Please consult a healthcare provider for advice."

# --- CORRECTED FUNCTION FOR FOLLOW-UP QUESTIONS ---
def get_follow_up_response(question, disease_context, api_key):
    """
    AI Step 3: Answers follow-up questions about a specific disease with a focus on OTC treatments and general advice.
    This function now separates the AI task from the disclaimer task.
    """
    # CORRECTED PROMPT: The AI is told to focus ONLY on providing helpful, safe advice.
    system_prompt = (
        f"You are a cautious and helpful medical information assistant. The user has been identified as potentially having '{disease_context}'. "
        "Your task is to answer their follow-up questions about over-the-counter (OTC) treatments and general wellness advice for this condition. "
        "Your response MUST be safe, general, and cautious. Follow these rules strictly:\n"
        "1. **DO NOT** write a disclaimer in your response. The application will add it separately.\n"
        "2. **BE GENERAL**: Suggest general categories of OTC products (e.g., 'pain relievers', 'decongestants') instead of brand names.\n"
        "3. **NO DOSAGES**: Never suggest dosages, frequencies, or treatment durations.\n"
        "4. **PROMOTE SAFETY**: Remind the user to read product labels and consult a pharmacist about side effects or interactions.\n"
        "5. **ALWAYS** encourage professional medical advice as the best course of action."
    )
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": question}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if not raw_text.strip():
            return "The model did not provide a response. Please rephrase your question."

        # MANUALLY ADDING THE DISCLAIMER: The code, not the AI, is responsible for this.
        disclaimer = "This is not medical advice. Always consult a healthcare professional or pharmacist before starting any new treatment."
        final_response = f"{disclaimer}\n\n{raw_text}"
        return final_response

    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        st.error("Error getting follow-up response.")
        print(f"DEBUG (Follow-up): {e}")
        return "Sorry, I was unable to process that follow-up question."


# --- Function for Generative Q&A ---
def get_generative_response(user_query, api_key):
    """Gets a direct answer for the Q&A tab."""
    system_prompt = ( "You are a helpful medical information assistant. Provide a clear, detailed, and accurate answer to the user's question based on web search results. DO NOT include disclaimers. Provide only the direct answer.")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = { "contents": [{"parts": [{"text": user_query}]}],"tools": [{"google_search": {}}],"systemInstruction": {"parts": [{"text": system_prompt}]},}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not raw_text.strip(): return "The model generated an empty response. Please try asking your question differently.", []
        disclaimer = "The following is for informational purposes only and is not a substitute for professional medical advice. Always consult a healthcare provider for any health concerns."
        final_response = f"{disclaimer}\n\n{raw_text}"
        grounding_metadata = result.get("candidates", [{}])[0].get("groundingMetadata", {})
        sources = [{"uri": attr["web"]["uri"], "title": attr["web"]["title"]} for attr in grounding_metadata.get("groundingAttributions", []) if "web" in attr]
        return final_response, sources
    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"DEBUG (Q&A): {e}")
        return "Sorry, there was an error connecting to the AI. Please try again later.", []

# --- Streamlit App UI ---
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")
st.markdown("""<style>.main { background-color: #f0f2f6; }</style>""", unsafe_allow_html=True)
st.title("🩺 Medical Information Hub")

if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
    st.error("⚠️ Your Google API Key is missing! Please paste it into the `API_KEY` variable at the top of the Python script.")
    st.stop()

tab1, tab2 = st.tabs(["**Medical Q&A**", "**Symptom Checker**"])

with tab1:
    st.header("Ask a Medical Question")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("E.g., What are the symptoms of vitamin D deficiency?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching for information..."):
                response_text, sources = get_generative_response(prompt, API_KEY)
                full_response = response_text
                if sources: full_response += "\n\n---\n**Sources:**\n" + "\n".join(f"{i+1}. [{s['title']}]({s['uri']})" for i, s in enumerate(sources))
                st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab2:
    st.header("AI-Powered Symptom Checker")
    st.markdown("Describe your symptoms in your own words. The AI will analyze them and you can ask follow-up questions.")
    df = load_data()

    # Initialize session state for the symptom checker
    if 'analysis_summary' not in st.session_state:
        st.session_state.analysis_summary = None
    if 'disease_context' not in st.session_state:
        st.session_state.disease_context = None
    if 'follow_up_messages' not in st.session_state:
        st.session_state.follow_up_messages = []

    if df is not None:
        user_text = st.text_area("Please describe how you are feeling:", placeholder="e.g., I have a high fever, my head hurts, and I've been coughing a lot.", height=100)
        
        if st.button("Analyze My Symptoms"):
            if user_text:
                # Reset previous analysis and follow-up chat
                st.session_state.analysis_summary = None
                st.session_state.disease_context = None
                st.session_state.follow_up_messages = []
                
                with st.spinner("AI is analyzing your symptoms..."):
                    all_symptoms_list = sorted(df.columns[1:].tolist())
                    extracted_symptoms = extract_symptoms_from_text(user_text, all_symptoms_list, API_KEY)
                
                if extracted_symptoms:
                    predictions, recognized = find_disease(extracted_symptoms, df)
                    with st.spinner("Generating your summary..."):
                        summary = generate_symptom_summary(recognized, predictions, API_KEY)
                    
                    st.session_state.analysis_summary = summary
                    st.session_state.disease_context = ", ".join(predictions)
                else:
                    st.error("The AI could not recognize any symptoms from your description that match our dataset. Please try describing them differently.")
            else:
                st.warning("Please describe your symptoms in the text box above.")

        if st.session_state.analysis_summary:
            st.subheader("Analysis Summary")
            st.markdown(st.session_state.analysis_summary)
            
            st.subheader("Ask a Follow-up Question")
            st.markdown(f"You can now ask questions about **{st.session_state.disease_context}**.")

            for message in st.session_state.follow_up_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if follow_up_prompt := st.chat_input("e.g., What over-the-counter treatments can help?"):
                st.session_state.follow_up_messages.append({"role": "user", "content": follow_up_prompt})
                with st.chat_message("user"):
                    st.markdown(follow_up_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_follow_up_response(follow_up_prompt, st.session_state.disease_context, API_KEY)
                        st.markdown(response)
                st.session_state.follow_up_messages.append({"role": "assistant", "content": response})

