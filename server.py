import flwr as fl
import numpy as np

# Create a custom strategy that saves the model after averaging
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # 1. Let the normal FedAvg algorithm do the math
        aggregated_weights, metrics = super().aggregate_fit(server_round, results, failures)

        # 2. If the math was successful, save the array to a file!
        if aggregated_weights is not None:
            print(f"\nSUCCESS: Saving 64-bit global model for round {server_round}...\n")
            # Convert Flower parameters to numpy arrays
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_weights)
            # Save as a compressed numpy file (.npz)
            np.savez(f"global_model_round_{server_round}.npz", *ndarrays)

        return aggregated_weights, metrics

if __name__ == "__main__":
    print(" Starting 64-bit Federated Server on Kali VM (With Auto-Save)...")
    
    # Use our new custom saving strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1
    )

    # Let's just do 1 round this time so you don't have to wait 13 minutes!
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1), 
        strategy=strategy
    )
