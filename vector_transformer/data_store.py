import pickle
import os
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader

class DataStore(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        # load data pickle if exists
        if os.path.exists(self.data_path):
            file = open(self.data_path, "rb")
            data = pickle.load(file)
            file.close()
        return []

    def write_data(self) -> None:
        # Write the data to the pickle file
        file = open(self.data_path, "wb")
        pickle.dump(self.data_path, file)
        file.close()

    def add_log_entry(self, query_text: str, query_vector: List[float], 
                      positive_vector: List[float], negative_vector: List[float], 
                      metadata: Dict[str, Any]) -> None:
        entry = {
            "query_text": query_text,
            "query_vector": query_vector,
            "positive_vector": positive_vector,
            "negative_vector": negative_vector,
            "metadata": metadata
        }
        self.data.append(entry)

        # Write the data to the file per 100 entries
        if len(self.data) % 100 == 0:
            print(f"Writing {len(self.data)} entries to {self.data_path}")
            self.write_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    datastore = DataStore('path_to_data.json')
    datastore.add_log_entry("query text", [1, 2, 3], [1, 1, 1], [0, 0, 0], {"genre": "Sci-Fi"})
    data_loader = datastore.get_data_loader(batch_size=32)
