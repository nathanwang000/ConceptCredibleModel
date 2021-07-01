import torch
from torch.autograd import grad

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
    print(f"error: {error:.2E}", "passed" if error < 1e-8 else "not pass")

def linear_pred_loss(x, y):
    '''
    x: (n, d1)
    y: (n, d2)
    return linear least square error from x to y
    '''
    return ((y - x @ p_inverse(x) @ y)**2).sum()

if __name__ == '__main__':

    d_unknown = 200 # dimension of unknown concepts
    d_concepts = 108 # dimension of known concepts
    bs = 32

    c = (torch.rand(bs, d_concepts) > 0.5).float()
    u = torch.randn(bs, d) # unknown concepts
    u.requires_grad = True

    # use c to predict u linearly and get the l2 error
    e = linear_pred_loss(c, u)
    print(grad(e, u)[0].shape)

    test_p_inverse()
