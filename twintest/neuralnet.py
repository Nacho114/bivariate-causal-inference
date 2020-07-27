import torch
import stats

# Wrapper

class Net:

    def __init__(self, params=None):
        if not params:
            self.init_default()

        else:
            self.init_from_params(params)

    def init_default(self):
        default_params = {
            'norm' : False,
            'learning_rate' : 1e-1,
            'epochs' : 500,
            'H' : 100
        }
        self.init_from_params(default_params)
        

    def init_from_params(self, params):
        self.norm = params['norm']
        self.learning_rate = params['learning_rate']
        self.epochs = params['epochs']
        H = params['H']
        self.model = get_model(H=H)
        self.loss_fn = get_loss()

    def train(self, x, y):
        x_tensor = self.transform(x, norm=self.norm)
        y_tensor = self.transform(y, norm=False)
        train_model(x_tensor, y_tensor, self.model, self.loss_fn, self.epochs, self.learning_rate)

    def predict(self, x):
        x = self.transform(x) 
        y_pred = self.model(x)

        return to_numpy(y_pred).squeeze()

    def transform(self, x, norm=True):
        if norm:
            x = stats.normalize(x)

        x = to_tensor(x)

        return x


# casting   

def to_tensor(x):
    x_tensor = torch.from_numpy(x).view(-1, 1).to(torch.float32)
    return x_tensor

def to_numpy(x_tensor):
    return x_tensor.detach().numpy()

# general

def get_model(H=100, D_in=1, D_out=1):

    model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )

    return model


def get_loss():
    return torch.nn.MSELoss(reduction='sum')


def train_model(x, y, model, loss_fn=None, epochs=500, learning_rate=1e-3, print_out=False):

    if not loss_fn:
        loss_fn = get_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if print_out and t % 100 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()