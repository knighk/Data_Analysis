proc print data=save.train2 (obs=10);

var 
score 	score1--score19
first_time_homebuyer 	first_time_homebuyer1
insurance 	insurance1--insurance9
number_units 	number_units1--number_units3
occupancy_status 	 	occupancy_status1--occupancy_status2
DTI 	 	DTI1--DTI9
UPB 	 	UPB1--UPB14
LTV 	 	LTV1--LTV5
OIR 	 	OIR1--OIR9
property_state 	 	property_state1--property_state53
property_type		property_type1--property_type5
loan_purpose		loan_purpose1--loan_purpose2
orig_loan_term		orig_loan_term1--orig_loan_term2
number_borrowers		number_borrowers1
seller		seller1--seller22
servicer		servicer1--servicer24;
format _all_;

run;
