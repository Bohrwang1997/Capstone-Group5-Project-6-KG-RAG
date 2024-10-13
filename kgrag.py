import pandas as pd
import spacy
from spacy.matcher import Matcher
import networkx as nx
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.embeddings import HuggingFaceEmbeddings
import torch


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}  


def load_definitions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        definitions = file.readlines()
    return [definition.strip() for definition in definitions if definition.strip()]


def build_knowledge_graph(definitions):
    entity_pairs = []
    relations = []

    for definition in definitions:
        triple = get_relation(definition)
        if all(triple):  
            entity_pairs.append([triple[0], triple[2]])  
            relations.append(triple[1]) 

 
    kg_df = pd.DataFrame({'source': [i[0] for i in entity_pairs],
                          'target': [i[1] for i in entity_pairs],
                          'edge': relations})

  
    G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    return G


nlp = spacy.load('en_core_web_sm')


def get_relation(sent):

    if ' is ' in sent:
   
        parts = sent.split(' is ')
        subject = parts[0].strip() 
        object_ = parts[1].strip() 
        relation = 'is' 
        return (subject, relation, object_)
    else:
        return ("", "", "") 


def query_knowledge_graph(graph, subject, relation="is"):

    if subject in graph.nodes:
     
        neighbors = [target for _, target, data in graph.edges(subject, data=True) if data['edge'] == relation]
        if neighbors:
            return neighbors
    return []


class GraphRetriever:
    def __init__(self, graph):
        self.graph = graph

    def search(self, query):
     
        doc = nlp(query)
        subject = ""
        for token in doc:
            if token.dep_ == "nsubj":
                subject = token.text
                break
    
        results = query_knowledge_graph(self.graph, subject)
        if results:
        
            return [Document(page_content=f"{subject} is {result}") for result in results]
        return []


class T5Assistant:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def create_prompt(self, query, retrieved_info):
   
        prompt = f"""Explain the concept or answer the question in a detailed manner using simple words and examples.
        Instruction: {query}
        Relevant information: {retrieved_info}
        Output:
        """
        return prompt

    def reply(self, query, retrieved_info):
        prompt = self.create_prompt(query, retrieved_info)
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
        outputs = self.model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


definitions = load_definitions('ctx_pd.txt')
graph = build_knowledge_graph(definitions)


retriever = GraphRetriever(graph)
assistant = T5Assistant()


def generate_queries(definitions):
    queries = []
    for definition in definitions:
     
        queries.append(f"What is {definition}?")
        queries.append(f"How is {definition} used?")
        queries.append(f"What are the benefits of {definition}?")
        queries.append(f"What are the challenges of {definition}?")
        queries.append(f"How does {definition} differ from other concepts?")
    return queries


sample_definitions = [
    "accreditation", "assessment", "bachelor's degree", "campus",
    "certificate", "credits", "degree", "discipline", "enrollment",
    "higher education", "student", "tuition"
]


generated_queries = generate_queries(sample_definitions)

for query in generated_queries:
    retrieved_docs = retriever.search(query)
    retrieved_info = " ".join([doc.page_content for doc in retrieved_docs])
    reply = assistant.reply(query, retrieved_info)
    print(f"Query: {query}\nGenerated Reply:\n{reply}\n")
