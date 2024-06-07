import streamlit as st
import pandas as pd
import openai
import re
from cachetools import cached, TTLCache
import streamlit.components.v1 as components
import json


# Set your OpenAI API key
def get_azure_openai_credentials(model_name):
    """Retrieves OpenAI credentials based on the selected model from the configuration file.

    Args:
        model_name (str): Name of the model ("gpt-3.5" or "gpt4").

    Returns:
        tuple: A tuple containing the Azure endpoint and API key for the specified model.
    """

    with open("config.json", "r") as f:
        config_file = json.load(f)
    if model_name == "gpt-35-turbo-16k":
        endpoint = config_file["gpt3.5"]["endpoint"]
        key = config_file["gpt3.5"]["key"]
        return endpoint, key

    elif model_name == "gpt-4-32k":
        endpoint = config_file["gpt4"]["endpoint"]
        key = config_file["gpt4"]["key"]
        return endpoint, key

    else:
        raise ValueError(f"Invalid model name: {model_name}")


@st.cache_data
def generate_response(system_prompt, user_prompt, model, max_tokens=1028):
    try:
        if model not in ["gpt-35-turbo-16k", "gpt-4-32k"]:
            raise ValueError(
                "Invalid model specified. Supported models are 'gpt-35-turbo-16k' and 'gpt-4-32k'.")
        print("model_name_", model)
        # Azure OpenAI credential setup
        azure_endpoint, azure_key = get_azure_openai_credentials(model)
        openai.api_type = "azure"
        openai.api_version = "2024-02-15-preview"  # Update with the appropriate version if needed
        openai.api_base = azure_endpoint
        openai.api_key = azure_key

        if model == "gpt-35-turbo-16k":
            # Reduce the length of the user prompt
            user_prompt = user_prompt[:4010]

            # Reduce the length of the system prompt
            system_prompt = system_prompt[:2047]
            print("user_prompt", user_prompt)
            print("system_prompt", system_prompt)

            response = openai.ChatCompletion.create(
                engine="datamodeller",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )

            # Check if the response exceeds the token limit
            if response['usage']['total_tokens'] >= max_tokens:
                # Truncate the response to fit within the token limit
                response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"][:max_tokens]

            return response["choices"][0]["message"]["content"].strip()

        elif model == "gpt-4-32k":
            response = openai.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                engine="check",
                max_tokens=4096,
                temperature=0.4)

            return response["choices"][0]["message"]["content"].strip()


    except Exception as e:
        # Handle exceptions, you can customize this part based on your needs
        print(f"Error: {str(e)}")
        return "An error occurred while generating the response."


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


def extract_erd_code(markdown_text):
    """Extracts the ERD code block from the given Markdown text.

  Args:
      markdown_text: The Markdown text containing the ERD code.

  Returns:
      The extracted ERD code block (everything between '`mermaid' and the next '`'),
      or None if no ERD code is found.
  """
    match = re.search(r"`mermaid\n(.*?)\n`", markdown_text, flags=re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None


@st.cache_data
def generate_relationships_and_keys(dataset: dict, max_tokens=4096, temperature=0.9):
    # Azure OpenAI credential setup
    azure_endpoint, azure_key = get_azure_openai_credentials("gpt-4-32k")
    openai.api_type = "azure"
    openai.api_version = "2024-02-15-preview"  # Update with the appropriate version if needed
    openai.api_base = azure_endpoint
    openai.api_key = azure_key

    """Analyzes a dataset and identifies relationships and keys.

  Args:
      dataset: A dictionary where keys are filenames and values are dataframes.
      max_tokens: Maximum number of tokens for GPT-4 response (default: 4096).
      temperature: Temperature for controlling randomness in GPT-4 response (default: 0.9).

  Returns:
      String containing identified relationships and keys, or error message.
  """
    try:
        print("Dataset:", dataset)

        # Combine all dataframes
        combined_df = pd.concat(dataset.values())

        # Craft a comprehensive prompt for LLM
        prompt = [
            {"role": "system", "content": "You are an expert in data modeling and entity relationship analysis."},
            {"role": "user", "content": f"""
                Analyze the provided dataset and identify:
                1. The relationships between entities (columns), specifying their types (one-to-one, one-to-many, many-to-many).
                2. The primary key for each dataset within the files.

                The dataset is structured as a dictionary where the key is the filename and the value is the dataset (rows).

                Dataset:\n{dataset}
            """}
        ]

        # Send the prompt to GPT-4
        response = openai.ChatCompletion.create(
            engine="check",
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract relationships and keys from LLM's response
        relationships = response["choices"][0]["message"]["content"]  # Assume LLM provides both in a structured format
        print("Relationships", relationships)
        return relationships

    except Exception as e:
        return f"An error occurred: {str(e)}"


@st.cache_data
def generate_erd(data):
    """Generates markdown code for an ERD based on relationships and keys.

  Args:
      data: A dictionary containing dataset information.

  Returns:
      String containing the generated ERD code in markdown format, or error message.
  """
    # Azure OpenAI credential setup
    azure_endpoint, azure_key = get_azure_openai_credentials("gpt-4-32k")
    openai.api_type = "azure"
    openai.api_version = "2024-02-15-preview"  # Update with the appropriate version if needed
    openai.api_base = azure_endpoint
    openai.api_key = azure_key

    try:
        relationships_and_keys = generate_relationships_and_keys(data)
        print("Relationships and Keys:", relationships_and_keys)
        erd_example = '''
                 erDiagram
                    SHORTFALL {
                        string DAT_RIF
                        string COD_ABI
                        string COD_EXP
                        string COD_OPERAZ
                        string COD_FIL_OPERAZ
                        string DAT_RIF_COD_ABI_COD_EXP_COD_OPERAZ PK
                    }
                    EXEMPTIONS {
                        string COD_ABI
                        string COD_OPERAZ
                        string COD_FIL_OPERAZ
                        string COD_ABI_COD_OPERAZ PK
                    }
                    EXPOSURES {
                        string DAT_RIF
                        string COD_ABI
                        string COD_EXP
                        string COD_OPERAZ
                        string COD_FIL_OPERAZ
                        string DAT_RIF_COD_ABI_COD_EXP_COD_OPERAZ PK
                    }
                    SHORTFALL ||--o{ EXEMPTIONS : "has"
                    SHORTFALL ||--o{ EXPOSURES : "has"
                    EXEMPTIONS }o--o{ EXPOSURES : "indirectly related through SHORTFALL'''

        # Use OpenAI to generate ERD code based on information
        prompt = [
            {"role": "system",
             "content": "You are an expert in data modeling and experienced in creating Entity Relationship Diagrams (ERDs) using mermaid.js."},
            {"role": "user", "content": f"""
                Create markdown code for an Entity Relationship Diagram (ERD) using the mermaid.js library. The ERD should illustrate the following relationships and keys:

                {relationships_and_keys}

                Ensure the markdown code is clear, accurate, and compatible with mermaid.js. Focus on representing relationships correctly, including their types (one-to-one, one-to-many, many-to-many), and defining primary keys accurately.

                Here is an example of a well-structured ERD in mermaid.js format for reference:

                {erd_example}

                Use this example as a guide to format the ERD correctly based on the provided relationships and keys.
            """}
        ]

        response = openai.ChatCompletion.create(
            engine="check",
            messages=prompt,
            max_tokens=4096,
            temperature=0.7
        )

        # Extract the generated ERD code
        erd_content = response["choices"][0]["message"]["content"]
        print("ERD Content:", erd_content)
        erd = extract_erd_code(erd_content)  # Assuming extract_erd_code function exists
        print("ERD:", erd)

        return erd

    except Exception as e:
        return f"An error occurred in generate_erd_openai: {str(e)}"


@st.cache_data
def mermaid_chart(markdown_code):
    new_markdown_code = markdown_code.replace("mermaid", "")
    print("new_markdown_code", new_markdown_code)

    try:
        html_code = f"""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
        <style>
            .mermaid-container {{
                overflow: hidden;
                position: absolute;
            }}
            .mermaid {{
                /* Add your custom styles here */
                background-color: #f5f5f5;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 100px;
            }}
        </style>
        <div class="mermaid-container">
            <div class="mermaid">{new_markdown_code}</div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script src='https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js'></script>
        <script>
            mermaid.initialize({{ startOnLoad: true }});

            // Enable zooming with panzoom
            const container = document.querySelector('.mermaid-container');
            const mermaidDiv = document.querySelector('.mermaid');
            const panzoom = window.panzoom(mermaidDiv);

            container.addEventListener('wheel', function (e) {{
                e.preventDefault();
                panzoom.zoomWithWheel(e);
            }});
        </script>
        """
        # print('htmlcode', html_code)
        return html_code
    except Exception as e:
        return f"An error occurred in mermaid_chart: {str(e)}"


# @st.cache_data(experimental_allow_widgets=True)
def business(model, metatag_system_prompt):
    try:
        if "content_generated" not in st.session_state:
            st.session_state.content_generated = False

        uploaded_tables = {}
        conversation_history = []

        st.title("Business View")
        st.sidebar.markdown("----")

        uploaded_files = st.sidebar.file_uploader(
            "Select the source code to interpret", accept_multiple_files=True
        )

        for uploaded_file in uploaded_files:
            code_txt = uploaded_file.getvalue()
            dataframe = pd.read_csv(uploaded_file)  # Read the entire dataset
            uploaded_tables[uploaded_file.name] = dataframe
            content = str(uploaded_file.name) + " " + str(code_txt)  # Include filename and content
            conversation_history.append({"role": "user", "content": content})

            # File preview code
            with st.expander(f"File Preview: {uploaded_file.name}"):
                st.table(dataframe)

        st.sidebar.markdown("----")

        # Predefined question set
        questions = {
            "Summary": "Give me a brief summary of the data uploaded in bullet points without mentioning the column "
                       "names?",
            "Use_Case": "Give me examples of potential use cases of these datasets?",
            "Relationships": "Are there any relationships within the columns of the data?",
            "Tabular Data": "Provide a table listing all column names, data types and description?",
        }

        storeResponses = ""
        qCount = 1
        relationshipResponse = ""

        if st.sidebar.button("Generate Contents") or st.session_state.content_generated:
            for q in questions:
                prompt = "\n".join([message["content"] for message in conversation_history])
                prompt += "\n" + questions[q]

                output = generate_response(metatag_system_prompt, prompt, model)
                storeResponses += f"Q{qCount}. {questions[q]}\n\n{output}\n\n\n\n"
                qCount += 1

                with st.expander(questions[q]):
                    st.write(output)
                    if q in ["README", "Code"]:
                        st.button(f"Download {q}")

                if q == "Relationships":
                    relationshipResponse = output

            # Display ERD
            entityDiagramCode = generate_erd(uploaded_tables)

            st.markdown("### Entity-Relationship Diagram (ERD)")
            if entityDiagramCode is not None:
                components.html(mermaid_chart(entityDiagramCode), width=500, height=600, scrolling=True)
            else:
                st.error("An error occurred while generating the ERD.")

            st.sidebar.download_button("Download Responses", data=storeResponses)
    except Exception as e:
        st.error(f"An error occurred in the business function: {str(e)}")


# @st.cache_data(experimental_allow_widgets=True) - THIS LINE BREAKS EVERYTHING, DAN LEAVE COMMENTED OUT
def tech(model, metatag_system_prompt):
    if "content_generated" not in st.session_state:
        st.session_state.content_generated = False

    conversation_history = []
    # conversation_history.append({"role": "assistant", "content": init_prompt})
    st.title("Technical View")
    st.write(
        "Please choose from the following options for your SQL generator needs, make use of either the quick access prompts or the chat bot feature. Upload your own documents to the left.")
    st.write(
        "Scegli tra le seguenti opzioni per le tue esigenze di generatore SQL, utilizza le istruzioni di accesso rapido o la funzione chat bot. Carica i tuoi documenti a sinistra.")
    st.sidebar.markdown("----")
    st.subheader("1. Quick Access Prompts")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    with col1:
        queryButton2 = st.button("Star Schema")
        st.write("1. Generating the Star Schema for the data with lineage")

    with col2:
        queryButton3 = st.button("Table Lineage")
        st.write("2. What is the lineage between the provided tables")

    with col3:
        queryButton4 = st.button("Data Model Visualization")
        st.write("3. Understand how your data can be structured in SQL tables with primary key relationships")
    with col4:
        queryButton5 = st.button("SQL Query Generate")
        st.write("4. Generate SQL queries relevant to the datasets uploaded.")
    # with col5:
    #   queryButton6 = st.button("Data Dictionary Example")
    # st.write("5. Example just using the loaded data dictionary")

    st.subheader("2. Chat Bot Prompt examples")
    st.write("Please use these examples in the chat bot to the left")

    with st.expander("Examples"):

        st.markdown('''




        Utilizzare il pulsante "Genera contenuto" per utilizzare le seguenti istruzioni preimpostate. Oppure utilizza "Ottieni codice SQL" con il tuo prompt personalizzato.

            •  SQL Star Schema: To understand how the following datasets can be organized into a star schema using fact table and dimension table:
                   "create a SQL schema based on the provided data, breaking it into meaningful tables with primary keys and also provide a tabular view of those tables."

            •  Viewing Tabular Data: For a detailed view of your schema or data in table format:
                   "Provide the tabular view of the above schema"

            • Generating SQL Code: To get the SQL code for creating tables with detailed column information:
                    "Can you show the data model in tabular format if we create several SQL tables based on this data with primary key relationships in details"

            • Data Model Visualization: To understand how your data can be structured in SQL tables with primary key relationships, use:
                   "Can you show all the column names, their datatypes in SQL format and brief description in a nice tabular format"

    ''')
    st.subheader("3. Generate all prompts")
    st.write("Or use the 'Generate Contents' button to generate all of the following preset prompts")
    # data_dictionary = pd.read_csv('table_schemaCSV.csv')
    # st.subheader('4. Dictionary')
    # with st.expander("Given Data"):
    #   st.dataframe(data_dictionary)
    #  data_dictionary_str = data_dictionary.to_string()

    query = st.sidebar.text_input(" Input your query to the Chat Bot")
    queryButton = st.sidebar.button("Generate Response")
    st.sidebar.markdown("----")
    st.sidebar.text("Generate all prompts:")
    allButton = st.sidebar.button("Generate Contents")

    Global_data_prompt = "DO NOT USE ANY DATA THAT IS NOT UPLOADED, if nothing is uploaded please reply ('Nothing has been uploaded)'"

    st.sidebar.markdown("----")

    uploaded_files = st.sidebar.file_uploader(
        "Select the source code to interpret", accept_multiple_files=True
    )

    for uploaded_file in uploaded_files:
        code_txt = uploaded_file.getvalue()
        content = str(uploaded_file.name) + " " + str(code_txt)
        dataframe = pd.read_csv(uploaded_file, nrows=20)  # Read only the first 20 rows
        conversation_history.append({"role": "user", "content": content})
        # st.write("filename:", uploaded_file.name)
        # st.table(dataframe)
        # st.code(code_txt.decode("utf-8"), language='python')
        with st.expander(f"File Preview: {uploaded_file.name}"):
            st.table(dataframe)

    st.header("Output")
    st.sidebar.markdown("----")

    # Predefined question set
    questions = {"""
        USING THE PROVIDED DATA or the """ +
                 "Star Schema": "Develop a star schema design for the provided data by: Identifying the appropriate fact "
                                "table(s) containing transaction records and measurable metrics."
                                "Selecting relevant dimension tables to provide context. Analyzing the "
                                "relationships between the fact and dimension tables, ensuring proper connections via foreign "
                                "keys or common fields. Proposing a star schema layout with the central fact table(s) and "
                                "directly linked dimension tables, illustrating their connections. Addressing scenarios with "
                                "multiple fact tables serving distinct analytical needs, and considering integrating them or "
                                "developing separate star schemas. Providing a practical example of the star schema design, "
                                "and a tabular representation",
                 "SQL table": "Create a SQL schema based on the PROVIDED DATA, breaking it into meaningful tables with primary "
                              "keys and also provide a tabular view of those tables. Please remember to provide lineage.",
                 "Table": "Provide the tabular view of the above schema",
                 "SQL code": "Provide the SQL code to create tables with the columns in the ACTUAL_COLUMN column in the data, "
                             "splitting the tables with assumed primary and foreign keys, provide some examples of the SQL queries that could be used.",
                 "Data Model": "Can you show the data model in tabular format if we create several SQL tables based on this "
                               "data with primary key relationships in detail?",
                 "Tabular Data": "Can you show all the column names, their datatypes in SQL format and a brief description in "
                                 "a nice tabular format?"""
                 }

    # for the above table -> the input to the 'get SQL code'
    storeResponses = ""
    qCount = 1
    if allButton or st.session_state.content_generated:
        for q in questions:
            # Modify the prompt to include only the first 20 rows of the dataset
            prompt = "\n".join([message["content"] for message in conversation_history])
            prompt += "\n" + questions[q]
            prompt += "\n" + metatag_system_prompt + Global_data_prompt
            print(prompt)
            # print(prompt)
            output = generate_response(metatag_system_prompt, prompt, model)
            storeResponses += (
                    f"Q{qCount}. " + questions[q] + "\n\n" + output + "\n\n\n\n"
            )
            qCount += 1
            with st.expander(questions[q]):
                st.write(output)
                if q in ["README", "Code"]:
                    st.button("Download " + q)
        st.sidebar.download_button("Download Responses", data=storeResponses)

    if queryButton or st.session_state.content_generated:
        prompt = "\n".join([message["content"] for message in conversation_history])
        prompt += "\n" + query + """"USING THE PROVIDED DATA  """
        with st.expander("Generating Chat Bot response:"):
            st.write(generate_response(metatag_system_prompt, prompt, model))

        st.sidebar.download_button("Download Responses", data=storeResponses)

    if queryButton2 or st.session_state.content_generated:
        starPrompt = """USING ONLY THE PROVIDED DATA, IF NOTHING IS UPLOADED REPLY ("No Files Uploaded") OTHERWISE""" + """. Do the following: 1. Develop a 
        star schema design with lineage for banking analytics by Identifying the appropriate fact table(s) containing 
        transaction records and measurable metrics. Selecting relevant dimension tables to provide context. 2. 
        Analyzing the relationships between the fact and dimension tables, ensuring proper connections via foreign 
        keys or common fields. 3. Proposing a star schema layout with the central fact table(s) and directly linked 
        dimension tables, illustrating their connections. Addressing scenarios with multiple fact tables serving 
        distinct analytical needs, and considering integrating them or developing separate star schemas. 4. Providing 
        a practical example of the star schema design, including a diagram or tabular representation, specific to the 
        banking sector. 5. Suggesting a SQL schema example to showcase the proposed star schema design table. Show 
        the example schema in absolute full and comprehensive, no 'other metrics' or 'add more fields as necessary' 
        lines 6. 7. Please remember to provide lineage. 8. illustrate their connections using a flow chart or ascii. """

        prompt = "\n".join([message["content"] for message in conversation_history])
        prompt += "\n" + starPrompt + Global_data_prompt
        print(prompt)
        with st.expander("Generating SQL Star Schema...:"):
            st.write(generate_response(metatag_system_prompt, prompt, model))

        st.sidebar.download_button("Download Responses", data=storeResponses)

    if queryButton3 or st.session_state.content_generated:
        lineagePrompt = """USING THE PROVIDED DATA """ + """. What is the lineage 
        between these tables, please provide SQL schema breaking it into meaningful tables using fact tables and 
        dimensions table with primary keys AND illustrate their connections AND THEN provide a tabular 
        view of those tables. Please remember to provide lineage and discuss it. """

        prompt = "\n".join([message["content"] for message in conversation_history])
        prompt += "\n" + lineagePrompt + Global_data_prompt
        print(prompt)
        # print(data_dictionary)
        with st.expander("Generating Lineage...:"):
            st.write(generate_response(metatag_system_prompt, prompt, model))

        st.sidebar.download_button("Download Responses", data=storeResponses)

    if queryButton4 or st.session_state.content_generated:
        dataModelVisPrompt = """USING THE PROVIDED DATA  """ + """. Can you show all the 
        column names, their datatypes in SQL format and brief description in a nice tabular format AND illustrate 
        their connections using a flow chart or ascii. Please provide some example SQL queries and a example SQL 
        Schema using relevant dimensions and fact tables that can be made on this financial data """
        prompt = "\n".join([message["content"] for message in conversation_history])
        prompt += "\n" + dataModelVisPrompt + Global_data_prompt
        print(prompt)
        # print(data_dictionary)
        with st.expander("Generating Data Model Visualisation"):
            st.write(generate_response(metatag_system_prompt, prompt, model))

        st.sidebar.download_button("Download Responses", data=storeResponses)

    if queryButton5 or st.session_state.content_generated:
        sqlQueryPrompt = """USING THE PROVIDED DATA or the """ + """. Perform the following to 
        provide some relevant SQL query examples: 1. Propose a star schema layout with the central fact table(s) and 
        directly linked dimension tables. 2. Illustrate the connections between the tables within the star schema 
        using a clear diagram or tabular format or ascii diagram. 3. Design and provide SQL query examples to 
        maniupulate the on the data. Specify primary and foreign keys for these tables. 4. Suggesting a SQL query 
        example to showcase the proposed star schema design table. 5. Include a brief discussion on data lineage, 
        explaining how data flows and is transformed within this schema. """
        prompt = "\n".join([message["content"] for message in conversation_history])
        prompt += "\n" + sqlQueryPrompt + Global_data_prompt
        print(prompt)
        # print(data_dictionary)
        with st.expander("Generating SQL Queries"):
            st.write(generate_response(metatag_system_prompt, prompt, model))

        st.sidebar.download_button("Download Responses", data=storeResponses)