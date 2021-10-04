import torch
from torch.autograd import grad

def EYE(r, x):
    '''
    expert yielded estimation
    r: risk factors indicator (d,)
    x: attribution (d,)
    
    '''
    assert r.shape == x.shape
    l1 = (x * (1-r)).abs().sum()
    l2sq = (r * x).dot(r * x)
    return  l1 + torch.sqrt(l1**2 + l2sq)

def wL2(r, x):
    '''
    addtional penalty to (1-r)
    r: risk factors indicator (d,)
    x: attribution (d,)
    '''
    assert r.shape == x.shape    
    return (x * (1-r)).dot(x * (1-r))

def wL1(r, x):
    '''
    addtional penalty to (1-r)
    r: risk factors indicator (d,)
    x: attribution (d,)
    '''
    assert r.shape == x.shape    
    return (x * (1-r)).abs().sum()

def p_inverse(x, eps=1e-10):
    '''
    pseudo inverse of x
    eps: small constant to makes sure (x.T @ x) is invertible
    '''
    return (x.T @ x + eps * torch.eye(x.shape[1])).inverse() @ x.T

def test_p_inverse():
    d = 10
    X = torch.rand(32, d)
    W = torch.rand(d, 200)
    y = X @ W
    error = ((y - X @ p_inverse(X) @ y)**2).max()
    print(f"error: {error:.2E}, 'r2': {r_sq(X, y).item():.2f},",
          "passed" if error < 1e-8 else "not pass")

def cosine_loss(x, y):
    '''
    x: (n, d1)
    y: (n, d2)
    make sure columns of y is orthogonal to columns of x
    
    this is more efficient than r_sq b/c no inverse calculated
    '''
    x = x / torch.norm(x, dim=0)
    y = y / torch.norm(y, dim=0)
    return ((x.T @ y)**2).sum()
    
def r_sq(x, y, eps=1e-10):
    '''
    x: (n, d1)
    y: (n, d2)
    eps: avoid division by 0
    return the R^2 = 1 - SS_res / SS_tot value of the linear fit
    where SS_tot = sum((y-mean(y, 0))**2)
    larger the more predictive
    '''
    b = torch.zeros(x.shape[0], 1)
    x = torch.cat((x, b), 1) # add bias term
    SS_res = ((y - x @ p_inverse(x) @ y)**2).sum()
    SS_tot = ((y - y.mean(0))**2).sum() + eps
    return 1 - SS_res / SS_tot

if __name__ == '__main__':

    d_unknown = 200 # dimension of unknown concepts
    d_concepts = 2 # dimension of known concepts
    # the problem with R2 is that bs need to be > d_concepts, ow. it is always 1
    # because it is always exactly solvable; to salvage this, we may need to
    # enforce that no random few features can predict the output;
    # this can be alternatively implemented as orthongonality constraint
    bs = 32

    c = (torch.rand(bs, d_concepts) > 0.5).float()
    # W = torch.randn(d_concepts, d_unknown)
    # u = c @ W
    u = torch.randn(bs, d_unknown) # unknown concepts
    u.requires_grad = True

    # use c to predict u and get r^2
    r2 = r_sq(c, u)
    print(f'r2 for this example (n={bs}, d_concepts={d_concepts}) is {r2:.2E}')
    print(f'the gradient has shape {grad(r2, u)[0].shape}')
    
    test_p_inverse()
