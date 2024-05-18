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
def generate_relationships(dataset: dict, max_tokens=4096, temperature=0.9) -> str:
    # Azure OpenAI credential setup
    azure_endpoint, azure_key = get_azure_openai_credentials("gpt-4-32k")
    openai.api_type = "azure"
    openai.api_version = "2024-02-15-preview"  # Update with the appropriate version if needed
    openai.api_base = azure_endpoint
    openai.api_key = azure_key

def generate_relationships_and_keys(dataset: dict, max_tokens=4096, temperature=0.9):
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
            {"role": "system", "content": "You are a knowledgeable data modeler and relationship expert."},
            {"role": "user",
             "content": f"""Please analyze the following dataset and identify:
                    1. Relationships between entities(all the columns), including their types (one-to-one, one-to-many, many-to-many).
                    2. Primary key for each file's dataset information by using Dataset (a dictionary where key is filename and value are the dataset rows):\n{dataset}"""}
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
            {"role": "system", "content": "You are a helpful assistant with experience in data modelling."},
            {"role": "user",
             "content":
                 f"""Create markdown code for an Entity Relationship diagram using the mermaid.js library, 
                    showing the following relationships and keys:\n{relationships_and_keys}. Make sure it is a good, 
                    correct and effective markdown code that can be used with mermaid.js, do not include class 
                    markdown code. Use the below Markdown Code in Example as context on how to create one for the 
                    given relationships and keys Example: {erd_example} 
                   
                 """}]

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


# @st.cache_data(experimental_allow_widgets=True)
def tech(model, metatag_system_prompt):
    if "content_generated" not in st.session_state:
        st.session_state.content_generated = False

    conversation_history = []
    # conversation_history.append({"role": "assistant", "content": init_prompt})
    st.title("Technical View")
    st.sidebar.markdown("----")
    st.markdown('''
   
   
    Please use the "Generate Contents button" to use the following preset prompts. Or use "Get SQL Code" with your own personalised prompt. 

Utilizzare il pulsante "Genera contenuto" per utilizzare le seguenti istruzioni preimpostate. Oppure utilizza "Ottieni codice SQL" con il tuo prompt personalizzato.
    
1. **SQL Table Creation**: To create a SQL schema based on provided data, use the prompt: 
   - "create a SQL schema based on the above data, breaking it into meaningful tables with primary keys and also provide a tabular view of those tables."
2. **Viewing Tabular Data**: For a detailed view of your schema or data in table format, use:
   - "Provide the tabular view of the above schema"

3. **Generating SQL Code**: To get the SQL code for creating tables with detailed column information, use:
   - "Can you show the data model in tabular format if we create several SQL tables based on this data with primary key relationships in details"

4. **Data Model Visualization**: To understand how your data can be structured in SQL tables with primary 
   key relationships, use:
   - "Can you show all the column names, their datatypes in SQL format and brief description in a nice tabular format"
   
5. **SQL Star Schema**": To understand how the following datasets can be organized into a star schema using fact table and dimension table. Use:
   - "Star Schema"
''')
    query = st.sidebar.text_input("Input your query")
    queryButton = st.sidebar.button("Get SQL code")

    st.sidebar.markdown("----")

    uploaded_files = st.sidebar.file_uploader(
        "Select the source code to interpret", accept_multiple_files=True
    )

    for uploaded_file in uploaded_files:
        code_txt = uploaded_file.getvalue()
        content = str(uploaded_file.name) + " " + str(code_txt)
        dataframe = pd.read_csv(uploaded_file, nrows=20)  # Read only the first 20 rows
        conversation_history.append({"role": "user", "content": content})
        st.write("filename:", uploaded_file.name)
        st.table(dataframe)
        # st.code(code_txt.decode("utf-8"), language='python')

    st.sidebar.markdown("----")

    # Predefined question set
    questions = {
        "SQL table": "Create a SQL schema based on the above data, breaking it into meaningful tables with primary "
                     "keys and also provide a tabular view of those tables.",
        "Table": "Provide the tabular view of the above schema",
        "SQL code": "Provide the SQL code to create tables with the columns in the ACTUAL_COLUMN column in the data, "
                    "splitting the tables with assumed primary and foreign keys",
        "Data Model": "Can you show the data model in tabular format if we create several SQL tables based on this "
                      "data with primary key relationships in detail?",
        "Tabular Data": "Can you show all the column names, their datatypes in SQL format and a brief description in "
                        "a nice tabular format?",
        "Star Schema": "Objective: Develop a comprehensive Star schema for banking analytics. This schema should "
                       "effectively organize transactional data and related dimensions to support efficient querying "
                       "and insights extraction. Fact Table Identification: Begin by reviewing the dataset to "
                       "identify tables that can serve as fact tables. Look for tables containing transaction "
                       "records, which are central to banking analytics. These tables should include measurable "
                       "metrics like transaction amounts, interest rates, and fee charges. Ensure these tables "
                       "contain or are linked via foreign keys to other relevant tables, facilitating a connection to "
                       "various dimensions.Dimension Table Selection: Identify tables that will serve as dimension "
                       "tables. These are crucial for providing context to our fact data, including customer "
                       "demographics, account types, product details, and temporal aspects. Dimension tables add "
                       "depth to the analysis, enabling detailed segmentation and trend identification.Relationship "
                       "Analysis: Carefully examine how each potential fact table relates to the identified dimension "
                       "tables. This step involves scrutinizing foreign key relationships and matching field names to "
                       "ensure a logical and business-relevant connection between tables. The goal is to directly "
                       "link each dimension table to at least one fact table, creating a star-like structure centered "
                       "around each fact table. "
                       "Star Schema Layout Proposal: Propose a detailed Star schema configuration, highlighting the "
                       "chosen central fact table(s) and their directly linked dimension tables. Provide clear "
                       "examples of foreign key relationships or common fields that facilitate these connections. "
                       "This layout should reflect the intricacies of banking operations and analytics requirements. "
                       "Complex Data Structure Handling: Address scenarios with multiple fact tables serving distinct "
                       "analytical functions (e.g., daily transactions versus long-term loan performance). Assess the "
                       "feasibility of integrating disparate data into a unified fact table or the necessity of "
                       "developing separate, related Star schemas or a more complex snowflake schema to accommodate "
                       "these varied analytical needs. Schema Example: Based on the dataset provided, outline an "
                       "example Star schema relevant to the banking sector. Include a diagram or a tabular "
                       "representation illustrating the fact table(s), dimension tables, and their interconnections, "
                       "ensuring it serves as a practical example of the Star schema principles discussed. "
                       "Deliverable: A well-structured Star schema design that enhances data accessibility and "
                       "analysis for banking operations, supported by examples and diagrams where applicable. This "
                       "schema should enable stakeholders to derive meaningful insights efficiently and support a "
                       "wide range of analytical applications within the banking sector. "
                       "FINALLY suggest a SQL schema example to showcase this."
    }

    # for the above table -> the input to the 'get SQL code'
    storeResponses = ""
    qCount = 1
    if st.sidebar.button("Generate Contents") or st.session_state.content_generated:
        for q in questions:
            # Modify the prompt to include only the first 20 rows of the dataset
            prompt = "\n".join([message["content"] for message in conversation_history])
            prompt += "\n" + questions[q]
            prompt += "\n" + metatag_system_prompt
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
        prompt += "\n" + query
        with st.expander("SQL code:"):
            st.write(generate_response(metatag_system_prompt, prompt, model))

        st.sidebar.download_button("Download Responses", data=storeResponses)
