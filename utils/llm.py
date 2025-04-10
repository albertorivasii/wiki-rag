from llama_cpp import Llama
from dotenv import load_dotenv


def load_mistral(model_path:str, n_gpu_layers:int=20, n_ctx:int=2048):
    """
    Load the Mistral model using LlamaCpp.
    
    Args:
        model_path (str): Path to the Mistral model file.
        n_gpu_layers (int): Number of GPU layers to use. Default is 20.
        n_ctx (int): Context size. Default is 2048.

    Returns:
        Llama: Loaded Mistral model.
    """
    return Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False
    )


def run_llm(llm:Llama, context:str, question:str) -> str:
    """
    Run the LLM with the given prompt and parameters.

    Args:
        model (Llama): Loaded Mistral model.
        prompt (str): Input prompt for the model.
        temperature (float): Sampling temperature. Default is 0.7.
        max_tokens (int): Maximum number of tokens to generate. Default is 512.

    Returns:
        str: Generated text from the model.
    """
    prompt= prompt= f"""
        <s>
        [INST]
        Answer the question based on the context provided below. If the answer is not in the context, say "I don't know".
        Context:
        {context}
        Question: {question}
        Answer:
        [\INST]
        """
    
    output= llm(prompt, temperature=0.7, top_p=0.95, max_tokens=512, stop=["<|/INST|>"])
    response= output["choices"][0]["text"]
    return response