from nnumpy import Parameter
import numpy as np

from nnumpy import Module, Container, LossFunction
from nnumpy.testing import gradient_check
from nnumpy.utils import to_one_hot
from nnumpy.utils import sig2col


class Linear(Module):
    """
    NNumpy implementation of a fully connected layer.

    Attributes
    ----------
    in_features : int
        Number of input features (D) this layer expects.
    out_features : int
        Number of output features (K) this layer expects.
    use_bias : bool
        Flag to indicate whether the bias parameters are used.

    w : Parameter
        Weight matrix.
    b : Parameter
        Bias vector.

    Examples
    --------
    >>> fc = Linear(10, 1)
    >>> fc.reset_parameters()  # init parameters
    >>> s = fc.forward(np.random.randn(1, 10))
    >>> fc.zero_grad()  # init parameter gradients
    >>> ds = fc.backward(np.ones_like(s))
    """
    
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # register parameters 'w' and 'b' here (mind use_bias!)
        # YOUR CODE HERE
        # raise NotImplementedError()

        # Register parameter w
        self.register_parameter('w', Parameter(np.empty((in_features, out_features))))
        if self.use_bias:
            # register parameter b
            self.register_parameter('b', Parameter(np.empty(out_features)))
        
        self.reset_parameters()
        
    def reset_parameters(self, seed: int = None):
        """ 
        Reset the parameters to some random values.
        
        Parameters
        ----------
        seed : int, optional
            Seed for random initialisation.
        """
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(size=self.w.shape)
        if self.use_bias:
            self.b = np.zeros_like(self.b)
    
    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, D) ndarray

        Returns
        -------
        s : (N, K) ndarray
        cache : ndarray or iterable of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        s = x @ self.w
        if self.use_bias:
            s = s + self.b

        cache = x
        return s, cache 
    
    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, K) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        dx : (N, D) ndarray
        """
        # YOUR CODE HERE
        # raise NotImplementedError()

        x = cache
        dx = grads @ self.w.T
        self.w.grad = x.T @ grads
        if self.use_bias:
            self.b.grad = np.sum(grads, axis=0)

        return dx

class Sequential(Container):
    """
    NNumpy module that chains together multiple one-to-one sub-modules.
    
    Examples
    --------
    Doubling a module could be done as follows:
    >>> module = Module()
    >>> seq = Sequential(module, module)
    
    Modules can be accessed by index or by iteration:
    >>> assert module is seq[0] and module is seq[1]
    >>> mod1, mod2 = (m for m in seq)
    >>> assert mod1 is module and mod2 is module
    """

    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and hasattr(modules[0], '__iter__'):
            modules = modules[0]
        
        for mod in modules:
            self.add_module(mod)

    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, D) ndarray

        Returns
        -------
        y : (N, K) ndarray
        cache : ndarray or iterable of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()

        caches = []
        for module in self._modules:
            x, cache = module.compute_outputs(x)
            # save the caches
            caches.append(cache)
        
        return x, caches

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, K) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        dx : (N, D) ndarray
        """
        # YOUR CODE HERE
        # raise NotImplementedError()

        for module, cache in zip(reversed(self._modules), reversed(cache)):
            grads = module.compute_grads(grads, cache)

        return grads
    
class LogitCrossEntropy(LossFunction):
    """
    NNumpy implementation of the cross entropy loss function
    computed from the logits, i.e. before applying the softmax nonlinearity.
    """

    def raw_outputs(self, logits, targets):
        """
        Computation of loss without reduction.

        Parameters
        ----------
        logits : (N, K) ndarray
        targets : (N, K) ndarray
        
        Returns
        -------
        cross_entropy : (N, ) ndarray
        cache : ndarray or iterable of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        # compute the softmax of the logits in a numerically stable way
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # compute the cross-entropy loss
        cross_entropy = -np.sum(targets * np.log(probs + 1e-8), axis=1)
        
        # cache the necessary values for the backward pass
        cache = (probs, targets)
        
        return cross_entropy, cache

    def raw_grads(self, grads, cache):
        """
        Computation of gradients for loss without reduction.

        Parameters
        ----------
        grads : (N, ) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        dlogits : (N, K) ndarray
        dtargets : (N, K) ndarray
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        # retrieve the cached values
        probs, targets = cache
        
        # compute the gradients w.r.t. the logits
        dlogits = probs - targets
        dlogits *= grads[:, np.newaxis]
        
        # compute the gradients w.r.t. the targets
        dtargets = -np.log(probs + 1e-8) * grads[:, np.newaxis]
        
        return dlogits, dtargets

def multi_channel_convolution2d(x, k):
    """
    Compute the multi-channel convolution of multiple samples.
    
    Parameters
    ----------
    x : (N, Ci, A, B)
    k : (Co, Ci, R1, R2)
    
    Returns
    -------
    y : (N, Co, A', B')
    
    See Also
    --------
    sig2col : can be used to convert (N, Ci, A, B) ndarray 
              to (N, Ci, A', B', R1, R2) ndarray.
    """
    # YOUR CODE HERE
    # raise NotImplementedError()

    # Get dimensions
    N, Ci, A, B = x.shape
    Co, _, R1, R2 = k.shape

    # Use sig2col to transform x
    x_col = sig2col(x, (R1, R2))
    
    # Reshape x_col to (N, A', B', Ci * R1 * R2)
    A_prime, B_prime = x_col.shape[2:4]
    x_col = x_col.reshape(N, A_prime, B_prime, -1)

    # Reshape k for matrix multiplication
    k_reshaped = k.reshape(Co, -1).T

    # Perform matrix multiplication and reshape the result
    y = np.dot(x_col, k_reshaped).reshape(N, Co, A_prime, B_prime)

    return y

class Conv2d(Module):
    """ Numpy DL implementation of a 2D convolutional layer. """
    
    def __init__(self, in_channels, out_channels, kernel_size, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        
        # create parameters 'w' and 'b'
        # YOUR CODE HERE
        # raise NotImplementedError()

        # Initialize parameters 'w' and 'b'
        self.register_parameter('w', np.random.randn(out_channels, in_channels, *kernel_size))

        if self.use_bias:
            self.register_parameter('b', np.zeros(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self, seed: int = None):
        """ 
        Reset the parameters to some random values.
        
        Parameters
        ----------
        seed : int, optional
            Seed for random initialisation.
        """
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(size=self.w.shape)
        if self.use_bias:
            self.b = np.zeros_like(self.b)
        
    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, Ci, H, W) ndarray
        
        Returns
        -------
        feature_maps : (N, Co, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()

        feature_maps = multi_channel_convolution2d(x, self.w)
        if self.use_bias:
            feature_maps += self.b.reshape((1, -1, 1, 1))

        return feature_maps, (x, self.w)
    
    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, Co, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        
        Returns
        -------
        dx : (N, Ci, H, W) ndarray
        """

        x, w = cache  # Extract the input 'x' and weights 'w' from the cache

        # Calculate padding dimensions for input gradient
        pad_height = w.shape[2] - 1
        pad_width = w.shape[3] - 1

        # Pad the gradients
        grads_padded = np.pad(grads, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Transpose the weights for the convolution
        w_transposed = w.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]  # flip the kernel

        # Perform convolution to get the gradient with respect to the input
        dx = multi_channel_convolution2d(grads_padded, w_transposed)
        
        # Gradient w.r.t. weights
        x_col = sig2col(x, w.shape[2:]).reshape(-1, w.shape[1] * w.shape[2] * w.shape[3])
        grads_col = grads.transpose(1, 0, 2, 3).reshape(w.shape[0], -1)
        self.w.grad = np.dot(grads_col, x_col).reshape(w.shape)

        # Gradient w.r.t. bias
        if self.use_bias:
            self.b.grad = grads.sum(axis=(0, 2, 3))

        else:
            self.b.grad = None

        return dx
    
class MaxPool2d(Module):
    """ Numpy DL implementation of a max-pooling layer. """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = tuple(kernel_size)

    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, C, H, W) ndarray

        Returns
        -------
        a : (N, C, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()

        N, C, H, W = x.shape
        pool_h, pool_w = self.kernel_size

        # Calculate the output shape after pooling
        out_h = 1 + (H - pool_h) // pool_h
        out_w = 1 + (W - pool_w) // pool_w

        # Apply sig2col function to obtain window elements
        x_col = sig2col(x, self.kernel_size, stride=self.kernel_size)

        # Reshape the col array for max pooling
        x_col = x_col.reshape(N, C, pool_h * pool_w, out_h * out_w)

        # Perform max pooling
        max_idx = np.argmax(x_col, axis=2)
        out = np.max(x_col, axis=2)

        # Reshape the output
        out = out.reshape(N, C, out_h, out_w)

        cache = (x, max_idx, self.kernel_size)
        return out, cache

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, C, H', W') ndarray
        cache : ndarray or tuple of ndarrays

        Returns
        -------
        dx : (N, C, H, W) ndarray
        """
        # YOUR CODE HERE

        x, max_idx, (pool_h, pool_w) = cache
        N, C, H, W = x.shape
        _, _, out_h, out_w = grads.shape

        dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        # Correctly access the max_idx
                        idx = max_idx[n, c, i * out_w + j]
                        # Calculate the position in the original image
                        h_start = i * pool_h
                        w_start = j * pool_w
                        h_idx = idx // pool_w
                        w_idx = idx % pool_w
                        # Add the gradient to the position of the maximum value
                        dx[n, c, h_start + h_idx, w_start + w_idx] += grads[n, c, i, j]

        return dx
    
class Identity(Module):
    """ NNumpy implementation of the identity function. """
        
    def compute_outputs(self, s):
        """
        Parameters
        ----------
        s : (N, K) ndarray
        
        Returns
        -------
        a : (N, K) ndarray
        cache : ndarray or iterable of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        return s, s
    
    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, K) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        ds : (N, K) ndarray
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        return grads


class Tanh(Module):
    """ NNumpy implementation of the hyperbolic tangent function. """
        
    def compute_outputs(self, s):
        """
        Parameters
        ----------
        s : (N, K) ndarray
        
        Returns
        -------
        a : (N, K) ndarray
        cache : ndarray or iterable of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        a = np.tanh(s)
        return a, s
    
    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, K) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        ds : (N, K) ndarray
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        s = cache
        a = np.tanh(s)
        ds = grads * (1 - a**2)
        return ds

    
class AlgebraicSigmoid(Module):
    """ NNumpy implementation of an algebraic sigmoid function. """

    def compute_outputs(self, s):
        """
        Parameters
        ----------
        s : (N, K) ndarray
        
        Returns
        -------
        a : (N, K) ndarray
        cache : ndarray or iterable of ndarrays
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        a = s / np.sqrt(1 + s**2)
        return a, s

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, K) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        ds : (N, K) ndarray
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        a = cache
        ds = grads * (1 / (1 + a**2)**(3/2))
        return ds