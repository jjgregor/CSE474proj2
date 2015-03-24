# CSE474proj2
 Machine Learning Classification and Regression


CSE474/574 Introduction to Machine Learning
Programming Assignment 2
Classification and Regression
Due Date: April 10th 2015
Maximum Score: 100
Note A zipped file containing skeleton Python script files and data is provided. Note that for each problem,
you need to write code in the specified function withing the Python script file. Do not use any Python
libraries/toolboxes, built-in functions, or external tools/libraries that directly perform classi-
fication, regression, function fitting, or ridge regression. Using any external code will result in 0
points for the corresponding problem.
Evaluation We will evaluate your code by executing script.py file, which will internally call the problem
specific functions. Also submit an assignment report (pdf file) summarizing your findings. In the problem
statements below, the portions under REPORT heading need to be discussed in the assignment report.
Data Sets Two data sets are provided:
1. A 2D sample data set in the file “sample.pickle” along with the class assignment.
2. A medical data set is provided in the file “diabetes.pickle” along with the target assignment. The input
variables correspond to measurements (physical, physiological, and blood related) for a given patient
and the target variable corresponds to the level of diabetic condition in the patient. It contains:
• xtrain (242 × 64) and ytrain (242 × 1) for training.
• xtest (200 × 64) and ytest (200 × 1) for testing.
Submission You are required to submit a single file called proj2.zip using UBLearns.
File proj2.zip must contain 2 files: report and script.py.
• Submit your report in a pdf format. Please indicate the team members, group number, and your course
number on the top of the report.
• The code file should contain all implemented functions. Please do not change the name of the file.
Using UBLearns Submission: Continue using the groups that you created for programming
assignment 2. You should submit one solution per group through the groups page. If you want to
change the group, contact the instructors.
Project report: The hard-copy of report will be collected in class at due date. The problem descriptions
specify what is needed in the report for each problem.
1
Problem 1 (10 code + 10 report = 20 points) Experiment with
Gaussian discriminators
Implement Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). Refer
to Lecture 20 slides and handouts. Implement two functions in Python: ldaLearn and qdaLearn which take
a training data set (a feature matrix and labels) and return the means and covariance matrix (or matrices).
Implement two functions ldaTest and qdaTest which return the true labels for a given test data set and
the accuracy using the true labels for the test data. The format of arguments and the outputs is provided
in the base code.
REPORT 1.
Train both methods using the sample training data (sample train). Report the accuracy of LDA and QDA
on the provided test data set (sample test). Also, plot the discriminating boundary for linear and quadratic
discriminators. Explain why there is a difference in the two boundaries.
Problem 2 (5 code + 5 report = 10 Points): Experiment with
Linear Regression
Implement ordinary least squares method to estimate regression parameters by minimizing the squared loss.



Note that this is same as maximizing the log-likelihood in the Bayesian setting. You need to implement the
function learnOLERegression. Also implement the function testOLERegression to apply the learnt weights
for prediction on both training and testing data and to calculate the root mean squared error (RMSE):



REPORT 2.
Calculate and report the RMSE for training and test data for two cases: first, without using an intercept
(or bias) term, and second with using an intercept. Which one is better?
Problem 3 (10 code + 10 report = 20 Points): Experiment with
Ridge Regression
Implement parameter estimation for ridge regression by minimizing the regularized squared loss as follows:



You need to implement it in the function learnRidgeRegression.
REPORT 3.
Calculate and report the RMSE for training and test data using ridge regression parameters using the the
testOLERegression function that you implemented in Problem 2. Use data with intercept. Plot the errors
on train and test data for different values of λ. Vary λ from 0 (no regularization) to 0.5 in steps of 0.001.
Compare the relative magnitudes of weights learnt using OLE (Problem 1) and weights learnt using ridge
regression. Compare the two approaches in terms of errors on train and test data. What is the optimal value
for λ and why?
2
Problem 4 (20 code + 5 report = 25 Points): Using Gradient Descent
for Ridge Regression Learning
As discussed in class, regression parameters can be calculated directly using analytical expressions (as in
Problem 2 and 3). To avoid computation of (X>X)
−1
, another option is to use gradient descent to minimize
the loss function (or to maximize the log-likelihood) function. In this problem, you have to implement the
gradient descent procedure for estimating the weights w.
You need to use the minimize function (from the scipy library) which is same as the minimizer that you
used for first assignment. You need to implement a function regressionObjVal to compute the regularized
squared error (See (3)) and its gradient with respect to w. In the main script, this objective function will
be used within the minimizer.
REPORT 4.
Plot the errors on train and test data obtained by using the gradient descent based learning by varying the
regularization parameter λ. Compare with the results obtained in Problem 3.
Problem 5 (10 code + 5 report = 15 Points): Non-linear Regression
In this problem we will investigate the impact of using higher order polynomials for the input features. For
this problem use the third variable as the only input variable:
x t r a i n = x t r a i n [ : , 3 ]
x t e s t = x t e s t [ : , 3 ]
Implement the function mapNonLinear.m which converts a single attribute x into a vector of p attributes,
1, x, x2
, . . . , xp
.
REPORT 5.
Using the λ = 0 and the optimal value of λ found in Problem 3, train ridge regression weights using the
non-linear mapping of the data. Vary p from 0 to 6. Note that p = 0 means using a horizontal line as the
regression line, p = 1 is the same as linear ridge regression. Compute the errors on train and test data.
Compare the results for both values of λ. What is the optimal value of d in terms of test error in each
setting? Plot the curve for the optimal value of p for both values of λ and compare.
Problem 6 (0 code + 10 report = 10 points) Interpreting Results
Using the results obtained for previous 4 problems, make final recommendations for anyone using regression
for predicting diabetes level using the input features.
REPORT 6.
Compare the various approaches in terms of training and testing error. What metric should be used to
choose the best setting?
3