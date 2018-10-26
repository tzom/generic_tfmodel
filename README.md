# generic_tfmodel

This repository contains all the ***common*** bits-and-pieces, which are needed to train and use a model (graph in tensorflow).
Primarily, it can be used as a template to quickly prototype and try ideas.

## Scope decorator

For example, the scope decorator cleans up the "variable.scope()" call. Allowing to quickly iterate through ideas without typing redundant tf-syntax. 

## Model skeleton/blueprint

model.py contains a generic model class.

It can be used as a template or a class to inherit from.
Running this script examplifies the training workflow for the MNIST dataset using a tiny network.