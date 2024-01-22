import torch
import torch.nn as nn
from data_store import DataStore

class AdjustableVectorTransformationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdjustableVectorTransformationModel, self).__init__()
        # Define the architecture of the network
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        # identity initialization
        for layer in self.layers:
            # Initialize weights to identity and biases to zero
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, adjustment_factor: float=1.):
        # Forward pass through the network
        transformed_vector = self.layers(x)
        # Adjust the transformation based on the adjustment_factor
        adjusted_vector = (1 - adjustment_factor) * x + adjustment_factor * transformed_vector
        return adjusted_vector
    
    def loss(self, x, transformed_vector, adjustment_factor, positive_vector, negative_vector):
        '''
        Compute the loss function for the model.
        We want the transformed vector to be close to the positive vector and far from the negative vector. 2 components L2 distances. 
        At the same time, we want the transformed vector to be close to the original vector when the adjustment factor is small. 3rd component.
        '''
        # Compute the loss for the positive vector
        positive_loss = torch.dist(transformed_vector, positive_vector, 2)
        
        # Compute the loss for the negative vector
        negative_loss = torch.dist(transformed_vector, negative_vector, 2)
        
        # Compute the loss for the original vector
        original_loss = torch.dist(transformed_vector, x, 2)
        
        # Compute the total loss
        loss = positive_loss + negative_loss + original_loss * (1-adjustment_factor)
        
        return loss

    def fit(self, data_loader, epochs:int=30, adjustment_factor:float=1.):
        self.train()
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # Iterate over epochs
        for ep in range(epochs):
            # Iterate over data
            ep_loss = []
            for batch in data_loader:
                # Get the batch data
                query_vector = batch["query_vector"]
                positive_vector = batch["positive_vector"]
                negative_vector = batch["negative_vector"]
                # Reset the gradients
                optimizer.zero_grad()
                # Forward pass through the network
                # print(query_vector.shape)
                # print(query_vector)
                transformed_vector = self(query_vector, adjustment_factor)
                # Compute the loss
                loss = self.loss(query_vector, transformed_vector, adjustment_factor, positive_vector, negative_vector)
                # Backpropagate the loss
                loss.backward()
                # Update the weights
                optimizer.step()
                # Print the loss
                ep_loss.append(loss.item())
            avg_loss = sum(ep_loss)/len(ep_loss)
            print("EP {:3}/{:<3} | loss: {:.5}".format(epochs, ep+1, avg_loss))
            # print(f"EP {ep} loss: {sum(ep_loss)/len(ep_loss)}")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def inference(self, query_vector, adjustment_factor):
        self.eval()
        return self(query_vector, adjustment_factor)


# EXAMPLE USAGE with dummy data store in data_store.py    
if __name__ == "__main__":
    # Load the data
    datastore = DataStore('dummy_data.pickle')
    
    # create 100 dummy log entry
    for i in range(100):
        datastore.add_log_entry(f"query text {i}", torch.tensor([1., 2., 3.]), torch.tensor([1., 1., 1.]), 
                                torch.tensor([0., 0., 0.]), {"genre": "Sci-Fi"})

    # Create the data loader
    data_loader = datastore.get_data_loader(batch_size=16)
    
    adjustment_factor = 1.

    # Define the model
    model = AdjustableVectorTransformationModel(input_dim=3, hidden_dim=10, output_dim=3)

    # test identity
    query_vector = torch.tensor([1., 2., 3.])
    transformed_vector = model(query_vector, adjustment_factor)
    print(query_vector)
    print(transformed_vector)
    print()

    # Train the model
    model.fit(data_loader, epochs=30, adjustment_factor=adjustment_factor)
    # Save the model
    # model.save("model.pt")
    # Load the model
    # model.load("model.pt")
    
    # Inference
    model.eval()
    transformed_vector = model.inference(query_vector, adjustment_factor)
    print(transformed_vector)
