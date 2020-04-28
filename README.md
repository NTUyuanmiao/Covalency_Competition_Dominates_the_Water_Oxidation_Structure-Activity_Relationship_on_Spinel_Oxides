# Covalency_Competition_Dominates_the_Water_Oxidation_Structure-Activity_Relationship_on_Spinel_Oxides
ML codes

The code in this repository shows how to use the established covalency competition model to predict the Max(DT,DO) of a given spinel oxide.



The excel file contains the features chosen to describe each element.
The pred file contains the codes of the covalency competition model.
The pred-without-structure-feature contains the codes of the trained model without structral features



There is one line of code (### Predict the performance of new materials ###) behind the pred.py file, remove all the comment symbols for all the following line code and replace
 "Add_The_Formatted_Data_Here” with the new materail's data. The predicted Max(DT,DO) of the new material will be printed. 

The data format of the new material shall be:
[composition ratio of the first element at a site, followed by the data in Table s7, then, fill in the data of the second element of a site and the two elements of b site using the same method]
if there is no element, fill in 0 instead.
for example:
The data format of [Zn0.5Co0.5]Co2O4 is: [0.5, 1.65, 88, 12, 2.3, 0.5, 1.88, 79, 9, 2.3, 2, 1.88, 68.5, 9, 2.3, 0, 0, 0, 0, 0]
If there are multiple materials, separate them using a comma, for example: [0.5, 1.65, 88, 12, 2.3, 0.5, 1.88, 79, 9, 2.3, 2, 1.88, 68.5, 9, 2.3, 0, 0, 0, 0, 0], [0.5, 1.61, 67.5, 3, 1.9, 0.5, 1.88, 79, 9, 2.3, 2, 1.88, 68.5, 9, 2.3, 0, 0, 0, 0] 

