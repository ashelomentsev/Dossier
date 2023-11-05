import json

import torch
from sentence_transformers import SentenceTransformer, util

from gen_label import generate_label


SIM_THRESHOLD = 0.7

def load_records(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(note_records, param):
    with open(param, "w") as f:
        json.dump(note_records, f)


def store_new_or_update(note_records, label_note):
    embedder = SentenceTransformer('all-mpnet-base-v2')
    encoded_label = embedder.encode(label_note["label"], convert_to_tensor=True)

    # save_json(label_note | {"embedding": encoded_label}, "./data/notes.json")

    # note_records_embeddings = [note["embedding"] for note in note_records]
    note_records_embeddings = embedder.encode([note["label"] for note in note_records], convert_to_tensor=True)

    cos_scores = util.cos_sim(encoded_label, note_records_embeddings)[0]
    top_results = torch.topk(cos_scores, k=1)
    top_sim_score = top_results[0][0].item()
    print('Top score:', top_sim_score)

    if top_sim_score > SIM_THRESHOLD:
        print('Found existing record:')
        updated_note = label_note["note"] + "\n\n" + note_records[top_results[1][0]]["note"]
        print('Updated note:', updated_note)
    else:
        print("No similar records found\nUpdating database with new record")
        # insert new record


def main():
    note_records = load_records("./data/records.json")
    # note_text = "I've met this guy at the Hackathon. He's Polish. I already don't remember his name. Kieran? Okay, his name is Kieran. He also has a second name. Thomas? He has knowledge and he was doing research in AI. And we've been working on a project together. I don't know much else about him, but he's wearing glasses. He has a beard. Nice, polite person."
    #
    # label_note = generate_label(note_text)
    label_note = load_records("./data/notes.json")[0]
    store_new_or_update(note_records, label_note)

    label_note = load_records("./data/notes.json")[1]
    store_new_or_update(note_records, label_note)


if __name__ == "__main__":
    main()
