from rgl.datasets.ogb import OGBRGLDataset
from rgl.utils import llm_utils
from openai import OpenAI
from rouge_score import rouge_scorer
from dotenv import load_dotenv
import os


def evaluate_abstracts(generated_abstract, ground_truth_abstract):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth_abstract, generated_abstract)
    return scores


# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

dataset = OGBRGLDataset("ogbn-arxiv")
titles = dataset.raw_ndata["title"]
abstracts = dataset.raw_ndata["abstract"]

query_node_indices = [0]

for qid, qnid in enumerate(query_node_indices):
    query_title = titles[qnid]
    query_abstract_gt = abstracts[qnid]

    prompt = f"""
You are an expert academic writer. Given the title of a query research paper, generate a concise, informative, and coherent abstract for the query paper.
Query Paper Title: {query_title}

Generate the abstract for the query paper below:"""

    print(f"=== Prompt Sent to {model} ===")
    print(prompt)
    print("=============================\n")

    sys_msg = "You are an expert in academic writing."
    generated_abstract = llm_utils.chat_openai(prompt, model=model, sys_prompt=sys_msg, client=client)
    
    print("=== Generated Abstract ===")
    print(generated_abstract)
    print("\n=== Ground Truth Abstract ===")
    print(query_abstract_gt)
    print("\n" + "=" * 80 + "\n")

    scores = evaluate_abstracts(generated_abstract, query_abstract_gt)
    print("ROUGE scores:")
    for key, value in scores.items():
        print(f"{key}: {value}")
