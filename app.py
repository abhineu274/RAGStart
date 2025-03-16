import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("ENDPOINT_URL", "https://aoai-d01.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Azure AI Search configuration
search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT", "https://your-search-service.search.windows.net")
search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
search_index_name = os.getenv("AZURE_AI_SEARCH_INDEX", "aj-aiindex")

embeddings_endpoint = os.getenv("AZURE_OPEN_AI_EMBEDDING_ENDPOINT")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-02-01"
)

# Streamlit app
st.title("Azure OpenAI + Azure AI Search Chatbot")
st.write("Enter your question below and click Submit to get a response.")

# Input field for the question
user_question = st.text_input("Your Question:")

# Submit button
if st.button("Submit"):
    if user_question.strip():
        try:
            # Corrected extra_body format for Azure AI Search integration
            extra_body = {
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "filter": None,
                            "endpoint": f"{search_endpoint}",
                            "index_name": f"{search_index_name}",
                            "semantic_configuration": "azureml_default",
                            "authentication": {
                                "type": "api_key",
                                "key": f"{search_api_key}"
                            },
                            "embedding_dependency":{
                                "type" : "endpoint",
                                "endpoint" : f"{embeddings_endpoint}",
                                "authentication" : {
                                    "type" : "api_key",
                                    "key" : f"{api_key}"
                                }
                            },
                            "query_type": "vector_simple_hybrid",  # Corrected field
                            # "fields": ["content"],  # Specify the text field for retrieval
                            "top_n_documents": 5  # Number of relevant results to return
                        }
                    }
                ]
            }

            # Generate response from Azure OpenAI with Azure AI Search
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use the retrieved data for better answers."},
                    {"role": "user", "content": user_question}
                ],
                extra_body=extra_body  # Pass corrected data source
            )

            # Display the response
            st.write("### Response:")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question before submitting.")
