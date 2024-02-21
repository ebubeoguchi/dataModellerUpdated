import streamlit as st
from helperFunctions import business, tech
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import json
import random
import time
import pandas as pd
import io
from dotenv import load_dotenv
import os
import csv

# Load variables from .env file
load_dotenv()

# to change from GPT4 to GPT4-Turbo
# in model selection
# choose: gpt-4-1106-preview

st.set_option('deprecation.showfileUploaderEncoding', False)

with open("config.json") as f:
    config = json.load(f)
    key_vault_url = config["KEY_VAULT_URL"]
    deployment_gpt35 = config["gpt3.5"]
    deployment_gpt4 = config["gpt4"]



# Set your OpenAI API key
openai.api_key = os.getenv('OPEN_AI_KEY')



# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
    st.session_state.uploaded_data = None  # Initialize uploaded_data variable


def generate_response(user_input, conversation_history, model, dataset):
    try:
        # Prompt modified to be more suitable for a chatbot
        prompt = f"User: {user_input}\nAssistant:"

        # Combine entire conversation history, including data, into a single string
        conversation_str = "\n".join([message["content"] for message in conversation_history])

        # Initialize messages with system and user prompts
        messages = [
            {"role": "system", "content": f"You are a helpful assistant designed to answer user questions. Please utilize the provided dataset ({dataset}) to generate responses. Ensure that your answers are in CSV format when referencing the dataset. Additionally, adhere to ethical guidelines and avoid providing sensitive or inappropriate information. If a user request falls outside the scope of the dataset or involves confidential information, politely refrain from providing an answer. Your primary goal is to assist users effectively while maintaining ethical standards and data privacy."},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": prompt},
        ]

        # Split the conversation_str into chunks dynamically based on the token limit
        max_tokens = 4096 - len(prompt)  # Adjust for the assistant prompt
        chunks = [conversation_str[i:i + max_tokens] for i in range(0, len(conversation_str), max_tokens)]

        # Generate response for each chunk of conversation
        for chunk in chunks:
            print('Chunk:', chunk)
            messages[-1]["content"] = chunk  
            print('Messages:', messages)
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )

            # Update assistant message content with the response
            messages[-1]["content"] = response["choices"][0]["message"]["content"].strip()

        return messages[-1]["content"]
    except Exception as e:
        st.error(f"An error occurred in generate_response: {str(e)}")



def main():
    try:
        metatag_system_prompt = """ 
            You are Meta Tag Pro, a data specialist and expert SQL Developer with a focus on risk management. In analyzing financial data from a bank, you are required to:
            - Examine the data with an emphasis on identifying potential risks, such as anomalies, unusual patterns, or signs of fraudulent activities. 
            - Always make some mention that this is a risk assesment tool. 
            - Describe the key components and structure of the data file, focusing on elements that are relevant to risk assessment.
             - Examine the data with an emphasis on identifying potential risks, such as anomalies, unusual patterns, or signs of fraudulent activities.
             - Describe the key components and structure of the data file, focusing on elements that are relevant to risk assessment.
            - Outline the data types of the values, the relationships between the data, and check if any dependencies are present in the data
            - Additionally, identify any potential data inconsistencies or abnormalities that you notice
            - Generate a SQL table schema and return only a SQL query based on user input to return data that matches said input
            
            - Take a breath. The user might interact with the tool in Italian or the data might contain Italian, please reply in English and Italian in brackets after
        """

        # detect language in the prompt
        # Please make it clear which headers- or give more additional context such as the file name extracted, data uploaded
        # make italian input  (from the input, user might type in italian)
        # 'this is appraoached from a risk point of view'
        #     make sure to pull before push (stay on dev branch)

        st.markdown(
            """
            <style>
                [data-testid=stSidebar] [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.sidebar:
            st.image("heraLogo.png", width=75)

        menu = ["Home", "Business View", "Technical View", "Chatbot"]
        choice = st.sidebar.selectbox("Select your role", menu)
        st.sidebar.markdown("----")

        display_model = st.sidebar.radio("Pick a model version", ("gpt-3.5-turbo", "gpt-4"))

        if display_model == "gpt-3.5-turbo":
            model = deployment_gpt35
        elif display_model == "gpt-4":
            model = deployment_gpt4

        if choice == "Home":
            home()
        elif choice == "Business View":
            business(model, metatag_system_prompt)
        elif choice == "Technical View":
            tech(model, metatag_system_prompt)
        elif choice == "Chatbot":
            chatbot(display_model)
    except Exception as e:
        st.error(f"An error occurred in main: {str(e)}")


def home():
    st.title("Data Modeller")
    st.markdown(
        """Data Modeller is a powerful tool that drives data management efficiency, assists with data analysis, providing SQL code by leveraging cutting-edge LLMs. \n
    • Easily understand existing datasets with structured, human-readable descriptions and automated metadata management. \n
    • Advanced capabilities include generating data product descriptions, creating data dictionaries and suggesting potential use cases and improvements, accelerating technical documentation creation. \n
    • Identify PII and sensitive information, providing an extra layer of governance to your data management processes. \n
    • For the technical user, the Data Assistant can provide SQL code based on plain english user input in proper code format from the data provided. \n
    • The Data Assistant can propose potential improvements in the structure/architecture of the data provided. \n
    """
    )

    st.header("Security and Compliance")
    st.markdown(
        "This Data Modeller works by leveraging Azure OpenAI services and its APIs. For use cases that are sensitive and highly regulated in nature, we have compiled the important security highlights of Azure OpenAI:"
    )
    st.markdown(
        "1. Data used to fine-tune models are stored in Azure Storage and are encrypted at rest. \n 2. User Prompts (including data uploaded from the UI) and its corresponding chat completions are stored in servers for 30 days, then deleted. \n 3. Access to this data are limited to Microsoft employees only in the case of Azure OpenAI service abuse by customer. \n 4. This 30 day data retention and Microsoft employee access can be removed by submitting a form to Microsoft defining the use-case. Once approved nothing will be retained in their servers. \n 5. Chat, completions, prompts are not used to train, test, retrain Azure OpenAI models \n 6. Currently, most of our prompts can produce good results from a well defined data dictionary, so redacting any further information from the data is being considered. \n"
    )
    st.markdown(
        "[Source](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy?context=%2Fazure%2Fcognitive-services%2Fopenai%2Fcontext%2Fcontext)"
    )



def load_data(files):
    try:
        data_list = []

        for file in files:
            # Load data from CSV or Excel file
            if file.name.endswith('.csv'):
                # Attempt to read CSV content using the csv module to detect delimiter
                content = file.read().decode('utf-8')
                dialect = csv.Sniffer().sniff(content)
                data_list.append(pd.read_csv(io.StringIO(content), dialect=dialect))
            elif file.name.endswith(('.xls', '.xlsx')):
                # For Excel files, load data from all sheets
                xls_data = pd.read_excel(file, sheet_name=None)

                for sheet_name, sheet_data in xls_data.items():
                    # Add a column for sheet name
                    sheet_data["SheetName"] = sheet_name
                    data_list.append(sheet_data)
            else:
                # For other file types, attempt to read as CSV after removing/replacing special characters
                st.warning(f"File {file.name} is not in CSV format. Attempting to convert to CSV.")
                content = file.read().decode('utf-8')

                # Check for common delimiters and proceed with the appropriate one
                for delimiter in [',', ';', '\t', '|']:
                    if delimiter in content:
                        data_list.append(pd.read_csv(io.StringIO(content), delimiter=delimiter))
                        break

        # Combine all loaded DataFrames into a single DataFrame
        data_combined = pd.concat(data_list, ignore_index=True)

        return data_combined
    except Exception as e:
        st.error(f"An error occurred in load_data: {str(e)}")


def chatbot(model):
    try:
        # Add a button to restart chat in the top right corner
        if st.button("Restart Chat", key="restart_chat_button", help="restart-chat-button"):
            st.session_state.messages = []  # Clear chat history
            st.session_state.uploaded_files = []  # Clear uploaded data

        st.title("Data Modeller Assistant")

        # Handle file uploading (moved outside the prompt input block)
        uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

        # Initialize conversation_history outside the for loop
        conversation_history = st.session_state.messages if "messages" in st.session_state else []

        if uploaded_files:
            # Process each uploaded file
            for idx, uploaded_file in enumerate(uploaded_files):
                # Load data from the uploaded file
                data = load_data([uploaded_file])

                # Display a message indicating that data has been uploaded
                st.success("Data has been successfully uploaded. You can now interact with the chatbot.")

                # Pass the dataset as context for the chatbot
                dataset_context = "\n".join(data.astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())

                # Accept user input
                if prompt := st.chat_input(f"What is up? ({idx})", key=f"chat_input_{idx}"):
                    # Add user message to conversation history
                    conversation_history.append({"role": "user", "content": prompt})

                    # Generate assistant response using the provided function
                    assistant_response = generate_response(prompt, conversation_history, model=model, dataset=data)

                    # Add assistant response to conversation history
                    conversation_history.append({"role": "assistant", "content": assistant_response})

                    # Display user message and assistant response in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)

                # Move data outside the for loop
                st.session_state.uploaded_data = data

            # Display the final chat history after processing all uploaded files
            st.session_state.messages = conversation_history

            # Flatten the nested list of conversation history for display
            unique_messages = []  # Initialize a list to store unique messages
            for message in st.session_state.messages:
                # Check if the message is not already in the unique_messages list
                if message not in unique_messages:
                    unique_messages.append(message)
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

    except Exception as e:
        st.error(f"An error occurred in chatbot: {str(e)}")


# def chatbot(model):
#     try:
#         # Add a button to restart chat in the top right corner
#         if st.button("Restart Chat", key="restart_chat_button", help="restart-chat-button"):
#             st.session_state.messages = []  # Clear chat history
#             st.session_state.uploaded_files = []  # Clear uploaded data

#         st.title("Data Modeller Assistant")

#         # Handle file uploading (moved outside the prompt input block)
#         uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

#         if uploaded_files:
#             conversation_history = []  # Initialize an empty list to store conversation history
#             # Process each uploaded file
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 # Load data from the uploaded file
#                 data = load_data([uploaded_file])

#                 # Display a message indicating that data has been uploaded
#                 st.success("Data has been successfully uploaded. You can now interact with the chatbot.")

#                 # Pass the dataset as context for the chatbot
#                 dataset_context = "\n".join(data.astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())

#                 # Accept user input
#                 if prompt := st.chat_input(f"What is up? ({idx})", key=f"chat_input_{idx}"):
#                     # Add user message to conversation history
#                     conversation_history.append({"role": "user", "content": prompt})

#                     # Generate assistant response using the provided function
#                     assistant_response = generate_response(prompt, conversation_history, model=model, dataset=data)

#                     # Add assistant response to conversation history
#                     conversation_history.append({"role": "assistant", "content": assistant_response})

#                     # Display user message and assistant response in chat message container
#                     with st.chat_message("user"):
#                         st.markdown(prompt)
#                     with st.chat_message("assistant"):
#                         st.markdown(assistant_response)

#                 # Move data outside the for loop
#                 st.session_state.uploaded_data = data

#             # Display the final chat history after processing all uploaded files
#             st.session_state.messages.append(conversation_history)

#             # Flatten the nested list of conversation history for display
#             flat_history = [message for sublist in st.session_state.messages for message in sublist]
#             unique_messages = []  # Initialize a list to store unique messages
#             for message in flat_history:
#                 # Check if the message is not already in the unique_messages list
#                 if message not in unique_messages:
#                     unique_messages.append(message)
#                     print("unique_messages", unique_messages)
#                     with st.chat_message(message["role"]):
#                         st.markdown(message["content"])

#     except Exception as e:
#         st.error(f"An error occurred in chatbot: {str(e)}")


# def chatbot(model):
#     try:
#         # Add a button to restart chat in the top right corner
#         if st.button("Restart Chat", key="restart_chat_button", help="restart-chat-button"):
#             st.session_state.messages = []  # Clear chat history
#             st.session_state.uploaded_files = []  # Clear uploaded data

#         st.title("Data Modeller Assistant")

#         # Handle file uploading (moved outside the prompt input block)
#         uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

#         conversation_history = []  # Initialize an empty list to store conversation history

#         if uploaded_files:
#             # Process each uploaded file
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 # Load data from the uploaded file
#                 data = load_data([uploaded_file])

#                 # Display a message indicating that data has been uploaded
#                 st.success("Data has been successfully uploaded. You can now interact with the chatbot.")

#                 # Pass the dataset as context for the chatbot
#                 dataset_context = "\n".join(data.astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())

#                 # Accept user input
#                 if prompt := st.chat_input(f"What is up? ({idx})", key=f"chat_input_{idx}"):
#                     # Add user message to conversation history
#                     conversation_history.append({"role": "user", "content": prompt})

#                     # Generate assistant response using the provided function
#                     assistant_response = generate_response(prompt, conversation_history, model=model, dataset=data)

#                     # Add assistant response to conversation history
#                     conversation_history.append({"role": "assistant", "content": assistant_response})

#                     # Display user message and assistant response in chat message container
#                     with st.chat_message("user"):
#                         st.markdown(prompt)
#                     with st.chat_message("assistant"):
#                         st.markdown(assistant_response)

#                 # Move data outside the for loop
#                 st.session_state.uploaded_data = data

#         # Display the final chat history after processing all uploaded files
#         st.session_state.messages.append(conversation_history)

#         # Flatten the nested list of conversation history for display
#         flat_history = [message for sublist in st.session_state.messages for message in sublist]
#         unique_messages = []  # Initialize a list to store unique messages
#         for message in flat_history:
#             # Check if the message is not already in the unique_messages list
#             if message not in unique_messages:
#                 unique_messages.append(message)
#                 with st.chat_message(str(message["role"])):
#                     st.markdown(str(message["content"]))


#     except Exception as e:
#         st.error(f"An error occurred in chatbot: {str(e)}")


def process_and_ask_questions(data, model):
    try:
        # Convert data to a string format for chatbot input
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False, sep=";")
        data_str = csv_buffer.getvalue()

        # Split the data_str into chunks to fit the model's maximum context length
        chunk_size = 4097
        data_chunks = [data_str[i:i + chunk_size] for i in range(0, len(data_str), chunk_size)]

        # Initialize messages with system and user prompts
        messages = [
            {"role": "system", "content": f"You are a helpful assistant"},
            {"role": "user", "content": "Interactively chat about the uploaded data."},
        ]

        # Generate response for each chunk of data
        for chunk in data_chunks:
            messages[-1]["content"] = chunk  # Update the user message content
            response = generate_response(chunk, messages[:-1], model=model, dataset=data_chunks)
            messages.append({"role": "assistant", "content": response})


        # Display chat messages for user input and assistant response
        st.session_state.messages.extend(messages)
    except Exception as e:
        st.error(f"An error occurred in process_and_ask_questions: {str(e)}")



# def chatbot(model):
#     try:
#         # Add a button to restart chat in the top right corner
#         if st.button("Restart Chat", key="restart_chat_button", help="restart-chat-button"):
#             st.session_state.messages = []  # Clear chat history
#             st.session_state.uploaded_files = []  # Clear uploaded data

#         st.title("Data Modeller Assistant")

#         # Handle file uploading (moved outside the prompt input block)
#         uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

#         if uploaded_files:
#             # Process each uploaded file
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 # Load data from the uploaded file
#                 data = load_data([uploaded_file])

#                 # Display a message indicating that data has been uploaded
#                 st.success("Data has been successfully uploaded. You can now interact with the chatbot.")

#                 # Pass the dataset as context for the chatbot
#                 dataset_context = "\n".join(data.astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())

#                 # Accept user input
#                 if prompt := st.chat_input(f"What is up? ({idx})", key=f"chat_input_{idx}"):
#                     # Add user message to chat history
#                     st.session_state.messages.append({"role": "user", "content": prompt})
#                     # Display user message in chat message container
#                     with st.chat_message("user"):
#                         st.markdown(prompt)

#                     # Generate assistant response using the provided function
#                     assistant_response = generate_response(prompt, st.session_state.messages, model=model, dataset=data)

#                     # Display assistant response in chat message container
#                     with st.chat_message("assistant"):
#                         st.markdown(assistant_response)

#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

#                 # Move data outside the for loop
#                 st.session_state.uploaded_data = data

#             # Display the final chat history after processing all uploaded files
#             final_messages = st.session_state.messages[-(idx + 1):]  # Display the latest conversation
#             print('final_message', final_messages)
#             for message in final_messages:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])

#     except Exception as e:
#         st.error(f"An error occurred in chatbot: {str(e)}")


# def chatbot(model):
#     try:
#         # Add a button to restart chat in the top right corner
#         if st.button("Restart Chat", key="restart_chat_button", help="restart-chat-button"):
#             st.session_state.messages = []  # Clear chat history
#             st.session_state.uploaded_files = []  # Clear uploaded data

#         st.title("Data Modeller Assistant")

#         # Handle file uploading (moved outside the prompt input block)
#         uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

#         if uploaded_files:
#             # Process each uploaded file
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 # Load data from the uploaded file
#                 data = load_data([uploaded_file])

#                 # Display a message indicating that data has been uploaded
#                 st.success("Data has been successfully uploaded. You can now interact with the chatbot.")

#                 # Pass the dataset as context for the chatbot
#                 dataset_context = "\n".join(data.astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())

#                 # Accept user input
#                 if prompt := st.chat_input(f"What is up? ({idx})", key=f"chat_input_{idx}"):
#                     # Add user message to chat history
#                     st.session_state.messages.append({"role": "user", "content": prompt})
#                     # Display user message in chat message container
#                     with st.chat_message("user"):
#                         st.markdown(prompt)

#                     # Generate assistant response using the provided function
#                     assistant_response = generate_response(prompt, st.session_state.messages, model=model, dataset=data)

#                     # Display assistant response in chat message container
#                     with st.chat_message("assistant"):
#                         st.markdown(assistant_response)

#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

#                 # Move data outside the for loop
#                 st.session_state.uploaded_data = data

#             # Display the final chat history after processing all uploaded files
#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])

#     except Exception as e:
#         st.error(f"An error occurred in chatbot: {str(e)}")



# def chatbot(model):
#     try:
#         # Add a button to restart chat in the top right corner
#         if st.button("Restart Chat", key="restart_chat_button", help="restart-chat-button"):
#             st.session_state.messages = []  # Clear chat history
#             st.session_state.uploaded_files = []  # Clear uploaded data

#         st.title("Data Modeller Assistant")

#         # Handle file uploading (moved outside the prompt input block)
#         uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

#         if uploaded_files:
#             # Display the chat history
#             for message in st.session_state.messages:
#                 if message["role"] == "user":
#                     with st.chat_message("user"):
#                         st.markdown(message["content"])
#                 elif message["role"] == "assistant":
#                     with st.chat_message("assistant"):
#                         st.markdown(message["content"])

#             # Process each uploaded file
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 # Load data from the uploaded file
#                 data = load_data([uploaded_file])

#                 # Display a message indicating that data has been uploaded
#                 st.success("Data has been successfully uploaded. You can now interact with the chatbot.")

#                 # Pass the dataset as context for the chatbot
#                 dataset_context = "\n".join(data.astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())

#                 # Display the chat history
#                 for message in st.session_state.messages:
#                     with st.chat_message(message["role"]):
#                         st.markdown(message["content"])

#                 # Accept user input
#                 if prompt := st.chat_input(f"What is up? ({idx})", key=f"chat_input_{idx}"):
#                     # Add user message to chat history
#                     st.session_state.messages.append({"role": "user", "content": prompt})
#                     # Display user message in chat message container
#                     with st.chat_message("user"):
#                         st.markdown(prompt)

#                     # Generate assistant response using the provided function
#                     assistant_response = generate_response(prompt, st.session_state.messages, model=model, dataset=data)

#                     # Display assistant response in chat message container
#                     with st.chat_message("assistant"):
#                         st.markdown(assistant_response)

#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

#                 # Move data outside the for loop
#                 st.session_state.uploaded_data = data
#     except Exception as e:
#         st.error(f"An error occurred in chatbot: {str(e)}")


# def process_and_ask_questions(data, model):
#     try:
#         # Convert data to a string format for chatbot input
#         csv_buffer = io.StringIO()
#         data.to_csv(csv_buffer, index=False, sep=";")
#         data_str = csv_buffer.getvalue()

#         # Split the data_str into chunks to fit the model's maximum context length
#         chunk_size = 4097
#         data_chunks = [data_str[i:i + chunk_size] for i in range(0, len(data_str), chunk_size)]

#         # Initialize messages with system and user prompts
#         messages = [
#             {"role": "system", "content": f"You are a helpful assistant"},
#             {"role": "user", "content": "Interactively chat about the uploaded data."},
#         ]

#         # Generate response for each chunk of data
#         for chunk in data_chunks:
#             messages[-1]["content"] = chunk  # Update the user message content
#             response = generate_response(chunk, messages, model=model)
#             messages.append({"role": "assistant", "content": response})

#         # Display chat messages for user input and assistant response
#         st.session_state.messages.extend(messages)
#     except Exception as e:
#         st.error(f"An error occurred in process_and_ask_questions: {str(e)}")




if __name__ == "__main__":
   main()
