Sugeno classifier
=================

Implementation of the Sugeno classifier, which was presented in the
paper "Machine Learning with the Sugeno Integral: The Case of Binary
Classification" `[1] <#1>`__.

Installation
============

Dependencies
------------

The Sugeno classifier requires: \* Python (>=3.8) \* NumPy (>=1.19.0) \*
scikit-learn (>=0.24.1) \* PuLP (>=2.4)

User installation
-----------------

Use the package manager `pip <https://pip.pypa.io/en/stable/>`__ to
install the Sugeno classifier.

.. code:: bash

    pip install sugeno-classifier

Usage
=====

The implementation is compatible to
`scikit-learn <https://scikit-learn.org/stable/>`__ and can be used like
other algorithms from this library. In order to use the Sugeno
classifier, import the class **SugenoClassifier** from the module
**sugeno\_classifier** from the package **classifier**. Some examples
are shown below:

First example
-------------

Use the contructor and the function **fit** to initialize the Sugeno
classifier for a given dataset.

.. code:: python

    >>> from classifier.sugeno_classifier import SugenoClassifier
    >>> X = [[1, 3, 2],
    ...      [2, 1, 3]]
    >>> y = [0, 1]
    >>> sc = SugenoClassifier()
    >>> sc.fit(X, y)

Use the function **predict** to classify samples.

.. code:: python

    >>> Z = [[3, 2, 1],
    ...      [1, 2, 3]]
    >>> sc.predict(Z)
    array([0, 1])

Example with hyperparameters
----------------------------

The Sugeno classifier has two hyperparameter, the maxitivity and the
margin, which can be set in the constructor. Both can influence the
classification performance. See `[1] <#1>`__ for more information.

.. code:: python

    >>> from classifier.sugeno_classifier import SugenoClassifier
    >>> X = [[1, 3, 2],
    ...      [2, 1, 3]]
    >>> y = [0, 1]
    >>> sc = SugenoClassifier(maxitivity=2, margin=0.01)
    >>> sc.fit(X, y)

Again, the function **predict** can be used to classify samples. Note
the different output compared to the first example.

.. code:: python

    >>> Z = [[3, 2, 1],
    ...      [1, 2, 3]]
    >>> sc.predict(Z)
    array([1, 1])

Example with different class labels
-----------------------------------

The classes do not have to be labeled with 0 and 1, they can be any
integer numbers or strings. The label, which is smaller in terms of the
relation < or lexicographically ordering, is assigned to negative class
and the other to the positive class.

The first example contains the class labels 2 and 4. Label 2 is assigned
to the negative class and label 4 is assigned to the positive class
because of 2<4.

.. code:: python

    >>> from classifier.sugeno_classifier import SugenoClassifier
    >>> X = [[1, 3, 2],
    ...      [2, 1, 3]]
    >>> y = [2, 4]
    >>> sc = SugenoClassifier()
    >>> sc.fit(X, y)
    >>> Z = [[3, 2, 1],
    ...      [1, 2, 3]]
    >>> sc.predict(Z)
    array([2, 4])

The second example contains the class labels 'one' and 'two'. Label
'one' is assigned to the negative class and label 'two' is assigned to
the positive class because 'one' comes lexicographically first.

.. code:: python

    >>> from classifier.sugeno_classifier import SugenoClassifier
    >>> X = [[1, 3, 2],
    ...      [2, 1, 3]]
    >>> y = ['one', 'two']
    >>> sc = SugenoClassifier()
    >>> sc.fit(X, y)
    >>> Z = [[3, 2, 1],
    ...      [1, 2, 3]]
    >>> sc.predict(Z)
    array(['one', 'two'])

License
=======

`MIT <https://choosealicense.com/licenses/mit/>`__

References
==========

[1] Sadegh Abbaszadeh and Eyke Hullermeier. Machine Learning with the
Sugeno Integral: The Case of Binary Classication. 2019.
