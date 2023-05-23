from rich import print, pretty
pretty.install()


#  Initial probabilities 

## Joint probabilities

print(" \n Part 1) Quantifying Uncertainty: \n")

P_A_s_B_s = 0.1

print("The joint probability of both Project A and B succeed is: %.2f" 
      % P_A_s_B_s)

P_A_s_B_f = 0.1

print("The joint probability of Project A succeed and B fail is: %.2f" 
      % P_A_s_B_f)

P_A_f_B_s = 0.1

print("The joint probability of Project A fail and B succeed is: %.2f" 
      % P_A_f_B_s)

P_A_f_B_f = 0.7

print("The joint probability of both Project A and B fail is: %.2f" 
      % P_A_f_B_f)

######################

print("\n")
# Marginal

P_A_s = 0.2

print("The probability of Project A succeeds is : %.2f" 
      % P_A_s)

P_A_f = 0.8
print("The probability of Project A fails is : %.2f" 
      % P_A_f)
####################

print("\n")

P_B_s_given_A_s = P_A_s_B_s / P_A_s

print("The probability of Project B succeeds given A succeeds is : %.2f" 
      % P_B_s_given_A_s)

P_B_f_given_A_s = P_A_s_B_f / P_A_s

print("The probability of Project B fails given A succeeds is : %.2f" 
      % P_B_f_given_A_s)

P_B_s_given_A_f = P_A_s_B_f / P_A_f

print("The probability of Project B succeeds given A fails is : %.2f" 
      % P_B_f_given_A_s)

P_B_f_given_A_f = P_A_f_B_f / P_A_f

print("The probability of Project B fails given A fails is : %.2f" 
      % P_B_f_given_A_f)


####################### Outcomes #############

print(" \n Part 2) Outcomes: \n")

V_A_s = 20
print("The net revenue if project A succeeeds (in $MM): %.2f" 
      % V_A_s)

V_A_f = -10
print("The net loss if project A fails (in $MM): %.2f" 
      % V_A_f)

V_B_s = 20
print("The net revenue if project B succeeds (in $MM): %.2f" 
      % V_B_s)


V_B_f = -10
print("The net loss if project B succeeds (in $MM): %.2f" 
      % V_B_f)

############################## Aletrenatives ###########

print(" \n Part 3) Evaluating the Alternatives: \n")

print("Alternative 1 is to develop only Project A") 

Alt_1 = P_A_s*V_A_s + P_A_f*V_A_f

print("The expected value of Alternative 1 (in $MM): %.2f" 
      % Alt_1)

print("\nAlternative 2 is to develop Project A and B simultanously") 

Alt_2 = P_A_s_B_s*(V_A_s+V_B_s) + P_A_s_B_f*(V_A_s+V_B_f) + \
    P_A_f_B_s *(V_A_f+V_B_s) + P_A_f_B_f *(V_A_f+V_B_f)
    
print("The expected value of Alternative 2 (in $MM): %.2f" 
      % Alt_2)

print("\nAlternative 3 is to develop Project A and B sequentially") 

Alt_3_A_s = max(P_B_s_given_A_s*(V_A_s+V_B_s) +  \
                P_B_f_given_A_s*(V_A_s+V_B_f) , 0)

Alt_3_A_f = max(P_B_s_given_A_f*(V_A_f+V_B_s) +  \
                P_B_f_given_A_f*(V_A_f+V_B_f) , 0)

Alt_3 = P_A_s*Alt_3_A_s + P_A_f*Alt_3_A_f

print("The expected value of Alternative 3 (in $MM): %.2f" 
      % Alt_3)

print("\nAlternative 4 is to do nothing") 

Alt_4 = 0
print("The expected value of Alternative 4 (in $MM): %.2f" 
      % Alt_4)
