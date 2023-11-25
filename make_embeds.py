from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm

import os

os.environ['TRANSFORMERS_CACHE'] = './cache/'


def get_embs(model_name: str, sentences: list[str], device: str, batch_size: int = 32) -> list[str]:
    embds = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]

        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.pooler_output

        embeddings = torch.nn.functional.normalize(embeddings).detach().cpu()

        for _ in embeddings:
            embds.append(_.numpy())

    return embds
