
from data import *
from utils import *
from MLP import MLP
from sgd import SGDMomentum


def main():
    random_seed(seed=seed, deterministic=True)
    # Create model, optimizer, and start training
    train_loader, val_loader = prepare_dataloaders(main_dataset)
    model_mlp = MLP(act_layer=nn.ReLU).to(device)
    optimizer = SGDMomentum(model_mlp.parameters(), lr=0.1) # you may tune lr
    loss_module = nn.CrossEntropyLoss().to(device)

    print(f'model mlp created: {count_parameters(model_mlp):05.3f}M')
    model_mlp=train_model(model_mlp, optimizer, loss_module, train_loader, val_loader, num_epochs=5, model_name="myMLP_ReLU")

    # Test best model on test set
    vanilla_mlp_test_acc = test_model(model_mlp, test_loader)
    print(f'Test accuracy: {vanilla_mlp_test_acc*100.0:05.2f}%')