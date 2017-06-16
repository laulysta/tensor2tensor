# Data generators for T2T models.

This directory contains data generators for a number of problems. We use a
naming scheme for the problems, they have names of the form
`[task-family]_[task]_[specifics]`.  Data for all currently supported problems
can be generated by calling the main generator binary (`t2t-datagen`). For
example:

```
t2t-datagen \
  --problem=algorithmic_identity_binary40 \
  --data_dir=/tmp
```

will generate training and development data for the algorithmic copy task -
`/tmp/algorithmic_identity_binary40-dev-00000-of-00001` and
`/tmp/algorithmic_identity_binary40-train-00000-of-00001`.
All tasks produce TFRecord files of `tensorflow.Example` protocol buffers.


## Adding a new problem

1. Implement and register a Python generator for the dataset
1. Add a problem specification to `problem_hparams.py` specifying input and
   output modalities

To add a new problem, you first need to create python generators for training
and development data for the problem. The python generators should yield
dictionaries with string keys and values being lists of {int, float, str}.
Here is a very simple generator for a data-set where inputs are lists of 1s with
length upto 100 and targets are lists of length 1 with an integer denoting the
length of the input list.

```
def length_generator(nbr_cases):
  for _ in xrange(nbr_cases):
    length = np.random.randint(100) + 1
    yield {"inputs": [1] * length, "targets": [length]}
```

Note that our data reader uses 0 for padding, so it is a good idea to never
generate 0s, except if all your examples have the same size (in which case
they'll never be padded anyway) or if you're doing padding on your own (in which
case please use 0s for padding). When adding the python generator function,
please also add unit tests to check if the code runs.

The generator can do arbitrary setup before beginning to yield examples - for
example, downloading data, generating vocabulary files, etc.

Some examples:

*   [Algorithmic generators](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/algorithmic.py)
    and their [unit tests](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/algorithmic_test.py)
*   [WMT generators](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/wmt.py)
    and their [unit tests](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/wmt_test.py)

When your python generator is ready and tested, add it to the
`_SUPPORTED_PROBLEM_GENERATORS` dictionary in the
[data
generator](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-datagen).
The keys are problem names, and the values are pairs of (training-set-generator
function, dev-set-generator function). For the generator above, one could add
the following lines:

```
  "algorithmic_length_upto100":
  (lambda: algorithmic.length_generator(10000),
   lambda: algorithmic.length_generator(1000)),
```

Note the lambdas above: we don't want to call the generators too early.
