ods graphics on;
ods html;
proc logistic data=save.train2;
model def_in_24_months1 = 
score1--score19
first_time_homebuyer1
insurance1--insurance9
number_units1--number_units3
occupancy_status1--occupancy_status2
DTI1--DTI9
UPB1--UPB14
LTV1--LTV5
OIR1--OIR9
property_state1--property_state53
property_type1--property_type5
loan_purpose1--loan_purpose2
orig_loan_term1--orig_loan_term2
number_borrowers1
seller1--seller22
servicer1--servicer24;
score data=save.valid2 out=save.scrtrain outroc=vroc;
roc; roccontrast;
run;

ods html close;
ods graphics off;
