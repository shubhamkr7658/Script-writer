import os
from typing import TypedDict

class pipelinestate(TypedDict):
    raw_input:str
    edited_text : str
    scripted_text : str
    final_output : str

   
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)


def editor_node(state:pipelinestate)->dict:
    prompt =(
        "you are an expert copyeditor.Clean up the following raw text. "
        "fix any grammatical errors,spelling mistakes, and smooth out transitic"
        "while keeping the core message intact.  Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response= llm.invoke(prompt)
    return {"edited_text": response.content.strip()}

def scriptwriter_node(state:pipelinestate) -> dict:
    prompt = (
        "You are an expert script writer. "
        "Convert the following edited text into a natural, engaging, and easy-to-understand script. "
        "Keep the original meaning intact while making it suitable for speaking in a video or presentation. "
        "Return only the script.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "scripted_text": response.content.strip()
    }

def translator_node(state: pipelinestate) -> dict:
    prompt = (
        "You are an expert translator. "
        "Translate the following script from English to Hindienglish. "
        "Preserve the original meaning, tone, and context. "
        "Use simple, natural, and hindienglish. "
        "Return only the translated text.\n\n"
        f"Script:\n{state['scripted_text']}"
    )
    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }

from langgraph.graph import StateGraph,START ,END
graph=StateGraph(pipelinestate)
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

graph.add_edge(START,"editor")
graph.add_edge('editor','scriptwriter')
graph.add_edge('scriptwriter','translator')
graph.add_edge('translator',END)

#compile
app=graph.compile() # app is ruable runable is invoke klar sakete hai

result=app.invoke({
    "raw_input": " ai is an agency . i think it ios very imp. it is alos future"
})
#output
print(result['final_output'])
