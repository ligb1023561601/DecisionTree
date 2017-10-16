# Decision Tree
## A decision tree based on ID3 algorithm  

* This project consists of two modules which are trees.py and tree_plot.py
* All the code is based ID3 algorithm that could not take advantage of the pruning stratege which means it can not deal  
  with overfitting.  
* The decision tree constructed by the algorithm can be stored into disk with the pickle tool of Python. 

## Usage ##

Call the following function in tree_plot.py to visualize the decision tree
	
`test_plot_tree()`  


Call the following function in trees.py to create a decision tree
	
`test_tree()`

Call the following function in trees.py to test the function for lens-buying
	
`test_glasses()`