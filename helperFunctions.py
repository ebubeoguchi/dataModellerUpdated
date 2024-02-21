import streamlit as st
import pandas as pd
import openai
import re
from cachetools import cached, TTLCache
import streamlit.components.v1 as components


@st.cache_data
def generate_response(system_prompt, user_prompt, model, max_tokens=1028):
    try:
        if model not in ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]:
            raise ValueError(
                "Invalid model specified. Supported models are 'gpt-3.5-turbo-0125' and 'gpt-4'.")

        if model == "gpt-3.5-turbo-0125":
            # Reduce the length of the user prompt
            user_prompt = user_prompt[:4010]

            # Reduce the length of the system prompt
            system_prompt = system_prompt[:2047]
            print("user_prompt", user_prompt)
            print("system_prompt", system_prompt)

            response = openai.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=0.5,
            )

            # Check if the response exceeds the token limit
            if response['usage']['total_tokens'] >= max_tokens:
                # Truncate the response to fit within the token limit
                response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"][:max_tokens]

            return response["choices"][0]["message"]["content"].strip()

        elif model == "gpt-4-turbo-preview":
            response = openai.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                max_tokens=4096,
                temperature=0.4 )

            return response["choices"][0]["message"]["content"].strip()


    except Exception as e:
        # Handle exceptions, you can customize this part based on your needs
        print(f"Error: {str(e)}")
        return "An error occurred while generating the response."


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

# @st.cache_data
def generate_erd(data, relationships):
    try:
        # Use OpenAI to generate ERD code based on relationships
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f"You are a data modeller. You have to create markdown code for Entity Relationship diagram for mermaid.js library using the following data info:{data} and data relationship info: {relationships}"},
            {"role": "assistant", "content": "Create the mermaid.js code for the Entity Relationship Diagram"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=prompt,
            max_tokens=4096,
            temperature=0.5
        )

        content = response["choices"][0]["message"]["content"]
        match = re.search(r"```mermaid(.*?)```", content, re.DOTALL)

        if match:
            print('Match!!!!!')
            erd_content = match.group(1)
            print('erd_content', erd_content)
            return erd_content
        else:
            return "No content found between triple single-quotes."

        # return erd_code
    except Exception as e:
        return f"An error occurred in generate_erd_openai: {str(e)}"

# @st.cache_data
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
                padding: 10px;
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



@st.cache_data(experimental_allow_widgets=True)
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
            "Tabular Data": "Provide a table listing all column names, data types, description, and PII information?",
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
            entityDiagramCode = generate_erd(uploaded_tables, relationshipResponse)

            st.markdown("### Entity-Relationship Diagram (ERD)")
            if entityDiagramCode is not None:
                components.html(mermaid_chart(entityDiagramCode), width=500, height=600, scrolling=True)
            else:
                st.error("An error occurred while generating the ERD.")

            st.sidebar.download_button("Download Responses", data=storeResponses)
    except Exception as e:
        st.error(f"An error occurred in the business function: {str(e)}")

@st.cache_data(experimental_allow_widgets=True)
def tech(model, metatag_system_prompt):
    if "content_generated" not in st.session_state:
        st.session_state.content_generated = False

    conversation_history = []
    # conversation_history.append({"role": "assistant", "content": init_prompt})
    st.title("Technical View")
    st.sidebar.markdown("----")

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
        "SQL table": "create a SQL schema based on the above data, breaking it into meaningful tables with primary keys and also provide a tabular view of those tables.",
        # 'Table': "Provide the tabular view of the above schema",
        # 'SQL code': 'Provide the SQL code to create tables with the columns in the ACTUAL_COLUMN column in the data splitting the tables with assumed primary and foreign keys',
        # 'Data Model': 'Can you show the data model in tabular format if we create several SQL tables based on this data with primary key relationships in details',
        "Tabular Data": "Can you show all the column names, their datatypes in SQL format, brief description and PII in a nice tabular format",
    }

    # for the above table -> the input to the 'get SQL code'
    storeResponses = ""
    qCount = 1
    if st.sidebar.button("Generate Contents") or st.session_state.content_generated:
        for q in questions:
            # Modify the prompt to include only the first 20 rows of the dataset
            prompt = "\n".join([message["content"] for message in conversation_history])
            prompt += "\n" + questions[q]

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
