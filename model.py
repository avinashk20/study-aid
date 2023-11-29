from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json
import networkx as nx
import matplotlib.pyplot as plt
import os


os.environ["OPENAI_API_KEY"] = "PUT YOUR API KEY"


class VideoProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )
        self.embeddings = OpenAIEmbeddings()

    def load_and_split(self, video_id):
        loader = YoutubeLoader(video_id=video_id)
        return loader.load_and_split(self.text_splitter)

    def get_summary(self, video_id, docs=None):
        if docs is None:
            docs = self.load_and_split(video_id)
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.run(docs)
        return summary, docs

    def get_answer(self, video_id, question, docs=None):
        if docs is None:
            docs = self.load_and_split(video_id)

        docsearch = Chroma.from_documents(docs, self.embeddings)

        qa_chain_prompt = PromptTemplate(
            template="The transcript for the video/lecture is provided in the square brackets. Based on the context answer the following question: [{question}], [{context}]. \n",
            input_variables=["question", "context"],
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )

        answer = qa.run(question)
        return answer, docs

    def get_question_answers(self, video_id, docs=None):
        if docs is None:
            docs = self.load_and_split(video_id)

        docsearch = Chroma.from_documents(docs, self.embeddings)

        class QuestionAnswer(BaseModel):
            questions: List[str] = Field(
                description="Questions that can be asked from this text."
            )
            answers: List[str] = Field(
                description="Answers to the corresponding questions."
            )

        parser = PydanticOutputParser(pydantic_object=QuestionAnswer)

        qa_chain_prompt = PromptTemplate(
            template="Use the following pieces of context to respond to the query at the end. {context}. \nFrom this text, formulate 5 items in the following format: \n{format_instructions}\n",
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )
        result = qa(
            {"query": "From this text, formulate 5 items in the following format:"}
        )
        result = json.loads(result["result"])
        return result, docs

    def get_mindmap(self, video_id: str, docs=None):
        if docs is None:
            docs = self.load_and_split(video_id)

        docsearch = Chroma.from_documents(docs, self.embeddings)

        class MindMap(BaseModel):
            main_title: str = Field("Main title/concept(< 3 words) of the video.")
            concepts: List[str] = Field(
                description="Major Concepts(< 3 words each) in the text"
            )
            subconcepts: List[List[str]] = Field(
                description="Corresponding list of sub-concepts(< 3 words each) for each concept with 3 subconcepts for each concept"
            )

        parser = PydanticOutputParser(pydantic_object=MindMap)

        qa_chain_prmpt = PromptTemplate(
            template="Use the following pieces of context to respond to the query at the end. {context}. \nFrom this text, formulate 4 items in following format: \n{format_instructions}\n",
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prmpt},
        )
        result = qa({"query": "From this text, formulate 4 items in following format:"})
        result = json.loads(result["result"])

        return self.get_graph(result), docs

    def get_graph(self, data_dict):
        print(data_dict)
        graph = nx.Graph()
        plt.clf()

        main_title = data_dict["main_title"]
        graph.add_node(main_title)

        concepts = data_dict["concepts"]
        for concept in concepts:
            graph.add_node(concept)
            graph.add_edge(main_title, concept)

        subconcepts = data_dict["subconcepts"][0:]
        for i, subconcept_list in enumerate(subconcepts):
            if i >= len(concepts):
                break
            for subconcept in subconcept_list:
                graph.add_node(subconcept)
                graph.add_edge(concepts[i], subconcept)

        layout = nx.spring_layout(graph, k=1, iterations=100)

        main_title_color = "red"
        center_concept_color = "orange"

        node_colors = (
            [main_title_color]
            + [center_concept_color] * len(concepts)
            + ["lightblue"]
            * sum(len(subconcept_list) for subconcept_list in subconcepts)
        )

        nx.draw_networkx(
            graph,
            pos=layout,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=10,
        )

        plt.axis("off")
        plt.savefig("mind_map.png", dpi=600, bbox_inches="tight")
        return plt.gcf()

    def get_notes(self, video_id, docs=None):
        if docs is None:
            docs = self.load_and_split(video_id)

        docsearch = Chroma.from_documents(docs, self.embeddings)

        qa_chain_prompt = PromptTemplate(
            template="Use the following pieces of context to generate structured lecture notes in markdown. I need notes in 2 parts: Important points and Important definitions. \n {context}. \n NOTES IN MARKDOWN: \n",
            input_variables=["context"],
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )
        result = qa({"query": "Formulate notes in Markdown:"})
        return result["result"], docs
