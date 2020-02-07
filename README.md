# Miscellaneous

### TASK DESCRIPTION

`Coding – Q1`

You are building a file sharing system intended to be used by ordinary people (e.g. your grandmother or other people with no computer skills whatsoever). 
One option in this system is to browse by filename. If people choose "sort by filename", then files should be sorted in the manner that helps them find the file they are looking for as quickly as possible. 
Here is an example list of files:
```
[ 'mallika_1.jpg', 'dog005.jpg', 'grandson_2018_01_01.png', 'dog008.jpg', 'mallika_6.jpg', 'grandson_2018_5_23.png', 'dog01.png', 'mallika_11.jpg', 'mallika2.jpg', 'grandson_2018_02_5.png', 'grandson_2019_08_23.jpg', 'dog9.jpg', 'mallika05.jpg' ]
```
Write a function that will sort this list of file names in the manner that the non-technical human user expects. 
Include with your answer both your code and also the result of running your code on the list of files above.
Requirements: You can use any programming language you like, but you must use the standard library only. 

Hint – Expected answer definitely involves more than a 1 line of code. 

* [Find Solution](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/CaseQ1.ipynb)

-------------------
`CaseStudy – Q2`

At Simpl, repayments are corner stone of our business model. We give credit to people in good faith and expect that they will repay back us in time. Thus it becomes imperative for us to track our repayments. One key step towards it is quick summary of repayments detailing out how much amount and number of people are pending or settled. We have a 15 day cycle and want to track this repayment info at a cycle level.
Make it as detailed as possible and make as many Metric as possible.

You have following data :
1. Settlement Data : Will help you with user info like status, amounts & time periods. This can tell you whether user settled after bill generation, before even bill generation or the user is still pending.
2. Cycles Data : Will help you map settlement info to cycles

What we expect:

Crisp summary of how we are doing each cycle
Feel free to ask as many questions as possible.
You are free to do this question in python or SQL.

Hint – Expected answer definitely involves more than a count.

-------------------

* [Find Solution](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/CaseStudyQ2-%20Complete%20Analysis%20and%20Visualization%20Report.ipynb)
--------------------

* Solution Images:

Output of Function `month_wise_billcreation_count`. It Shows the Frequency of Bill Creation on every Billing Cycle.

![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Bill%20Creation%20Frequency%20Distribution.png)
---------------
Output of Function `month_wise_billcreation_count`.It shows the Frequency of Bill Paying on every Billing Cycle.

![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Bill%20Paid%20Frequency%20Distribution.png)
--------------
Output of Function `total_bill_status_cyclewise`. It shows the total Amount of BillPaid and Created during each Billing period.

![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Billing%20Transactioon%20Details%20for%20Each%20Cycle.png)
--------------
Graph show Billing cycle of Total Amount of Bill Created and Paid.

![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Cycle%20wise%20Amount.png)
-------------

This is output of Function `user_id_performance_over_entire_billcycle`. This shows the behaviour of a particular user over all the billing periods.
![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Cycle%20wise%20Amount_userid.png)
-------------

Graph show Billing cycle of Total Count of Bill Created and Paid.
![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Cycle%20wise%20user_id%20Count.png)
-------------


This is output of Function `user_id_performance_over_entire_billcycle`. This shows the behaviour of a particular user over all the billing periods.
![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Cycle%20wise%20user_id%20Count_userid.png)
------------

This shows Top 20 user_id interaction frequency.

![image](https://github.com/99sbr/Miscellaneous/blob/master/Simpl/Assignment_Product_Analyst/Top%2020%20user_id%20interaction%20frequency.png)
-----------

