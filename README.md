# week_4KNN

Instructions:

Let's build a kNN model using the college completion data. 
The data is messy and you have a degrees of freedom problem, as in, we have too many features.  

You've done most of the hard work already, so you should be ready to move forward with building your model. 

1. Use the question/target variable you submitted and 
build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary). 

2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
if needed changed the target variable.

3. Create a dataframe that includes the test target values, test predicted values, 
and test probabilities of the positive class.

4. No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion matrix look the same at the same threshold 
levels or not? Why or why not?

5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
concerns or positive elements do you have about the model as it relates to your question? 

6. Create two functions: One that cleans the data & splits into training|test and one that 
allows you to train and test the model with different k and threshold values, then use them to 
optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
function just run them separately.) 

7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
step 7.
