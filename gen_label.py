from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

API_KEY = "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"

def generate_label(note_text):
    chat = ChatAnthropic(
        anthropic_api_key=API_KEY)
    template = ("You are a helpful personal assistant that helps to store informations about people I meet."
                "Analyse the following text and extract key information about the person."
                "Write the information in the structured format"
                "If some informations are missing do not write them."
                "For example:"
                "<example>"
                "Note: I met a guy called John, he lives here in London. He is a software engineer. "
                "He is 30 years old. He is married and has 2 kids. He likes to play football. He is a fan of Arsenal. "
                "Assistant: <name>John</name><age>30</age><city>London</city><job>software engineer</job><family>married, 2 kids</family><hobby>play football</hobby><interests>Arsenal</interests>"
                "</example>"
                "<example>"
                "Note: I met a woman, I dont know her name, but she is 25 years old. She is a doctor. She lives in Berkley."
                "Assistant: <age>25</age><city>Berkley</city><job>doctor</job>"
                "</example>"
                "<example>"
                "Note: I met a person called Tim in an old town of Dubrownik. He seems to be old, but I really like him. He can talk about architecture for hours. "
                "Assistant: <name>Tim</name><city>Dubrownik</city><hobby>architecture</hobby>"
                "</example>"
                "<example>"
                "Note: I met Alice in a bar. She is a student at the local University"
                "Assistant: <name>Alice</name><job>student</job><location>bar</location>"
                "")
    human_template = "Note: {text}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    messages = chat_prompt.format_messages(text=note_text)
    model_response_label = chat(messages)

    return {"label": model_response_label.content, "note": note_text}


def main():
    note_text = "I've met this guy at the Hackathon. He's Polish. I already don't remember his name. Kieran? Okay, his name is Kieran. He also has a second name. Thomas? He has knowledge and he was doing research in AI. And we've been working on a project together. I don't know much else about him, but he's wearing glasses. He has a beard. Nice, polite person."
    model_response = generate_label(note_text)

    print(model_response)


if __name__ == "__main__":
    main()
