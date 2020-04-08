## Object recognition using Spiking Neural Networks on SpiNNaker and GeNN

This repository contains an insect-inspired Spiking Neural Network (SNN) 
whose purpose is the correct recognition/classification of the presented input.
This is done through unsupervised learning and competition. The network is
expressed in the PyNN network description language and experiments are run 
with Graphics Processing Units (GPUs; through [GeNN](https://github.com/genn-team)) 
and the SpiNNaker neuromorphic system as compute backends. 

* The network consists of three layers (input, middle and output):
   * The _Input_ layer represents an encoding mechanism, in this case we are 
   using rank-order encoding of images through a procedure which resembles the
   computation in the mammalian retina.
   The code used to convert the images can be located 
   [here](https://github.com/chanokin/convert_to_roc). Since the conversion is
   slow, the encoded images are attached to PyNN SpikeSourceArrays
   
   * The _Middle_ layer is inspired by a region of insects' mushroom body. Its 
   purpose is to perform a dimensional expansion and sparsify the representation.
   To achieve this the probability of input connectivity is low (~10%), we added
   a distance constraint so that neurons in the _Middle_ layer can only connect 
   to nearby _Input_ neurons. Neurons who share input regions will compete through
   mutual inhibition (or a soft winner-takes-all [sWTA] per sub-population). 
   
   * The _Output_ layer is another region of insects' mushroom body. The main 
   purpose in our experiment is to have a readout region for the classification
   of the input patten. The synapses coming from the _Middle_ layer are modified
   through an unsupervised spike-timing-dependent plasticity (STDP) 
   rule, together with a sWTA circuit to promote specialization of _Output_ 
   neurons.

* To tune the network hyper-parameters we use the 
 [Learning-to-Learn framework](https://github.com/IGITUGraz/L2L) 
 developed by colleagues from the TU Graz and the Jülich Supercomputing Centre.
 
* Currently working on:
   * Juwels super-computer (GPU and SLURM)
   * HBP neuromorphic platform (SpiNNaker machine and MPI)
   * Desktop (GPU and MPI)
