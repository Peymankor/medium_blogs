from rich import print, pretty
pretty.install()


#  Initial probabilities 
#dsa
## Joint probabilities

#Test
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

######################



####################### Outcomes #############

V_A_s = 20

V_A_f = -5

V_B_s = 30

V_B_f = -10

############################## Aletrenatives ###########

Alt_1 = P_A_s*V_A_s + P_A_f*V_A_f

Alt_2 = P_A_s_B_s*(V_A_s+V_B_s) + P_A_s_B_f*(V_A_s+V_B_f) + \
    P_A_f_B_s *(V_A_f+V_B_s) + P_A_f_B_f *(V_A_f+V_B_f)
    
    
Alt_3_A_s = max(P_B_s_given_A_s*(V_A_s+V_B_s) +  \
                P_B_f_given_A_s*(V_A_s+V_B_f) , 0)

print(Alt_3_A_s)

Alt_3_A_f = max(P_B_s_given_A_f*(V_A_f+V_B_s) +  \
                P_B_f_given_A_f*(V_A_f+V_B_f) , 0)

print(Alt_3_A_f)

Alt_3 = P_A_s*Alt_3_A_s + P_A_f*Alt_3_A_f

print(Alt_3) 
 