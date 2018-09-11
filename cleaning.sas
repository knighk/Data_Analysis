proc format;
 value $missfmt ' '='Missing' other='Not Missing';
 value  missfmt  . ='Missing' other='Not Missing';
run;
ods printer pdf file = 'E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Reports\Missing Value\missing.pdf';
proc freq data=WORK.pos_cash_balance; 
Title "Pos_Cash_Balance Missing";
format _CHAR_ $missfmt.;
tables _CHAR_ / missing missprint nocum nopercent;
format _NUMERIC_ missfmt.;
tables _NUMERIC_ / missing missprint nocum nopercent;
run;
proc freq data=WORK.installments_payments;
Title 'Instalmens Payment Missing' 
format _CHAR_ $missfmt.;
tables _CHAR_ / missing missprint nocum nopercent;
format _NUMERIC_ missfmt.;
tables _NUMERIC_ / missing missprint nocum nopercent;
run;
proc freq data=WORK.credit_card_balance; 
Title "Credit Card Balance Missing";
format _CHAR_ $missfmt.;
tables _CHAR_ / missing missprint nocum nopercent;
format _NUMERIC_ missfmt.;
tables _NUMERIC_ / missing missprint nocum nopercent;
run;

proc freq data=WORK.previous_application; 
Title "Previous Application Missing";
format _CHAR_ $missfmt.;
tables _CHAR_ / missing missprint nocum nopercent;
format _NUMERIC_ missfmt.;
tables _NUMERIC_ / missing missprint nocum nopercent;
run;
proc freq data=WORK.application_train; 
Title "Application Train Missing";
format _CHAR_ $missfmt.;
tables _CHAR_ / missing missprint nocum nopercent;
format _NUMERIC_ missfmt.;
tables _NUMERIC_ / missing missprint nocum nopercent;
run;
ods printer pdf close;
/*Check the missing on pos cash balance*/
PROC UNIVARIATE DATA=WORK.POS_CASH_BALANCE;
var	CNT_INSTALMENT CNT_INSTALMENT_FUTURE;
HISTOGRAM CNT_INSTALMENT CNT_INSTALMENT_FUTURE;
RUN;

/*REPLACE ALL MISSING TO -1 IN POS_CASH_BALANCE*/
proc stdize data=work.pos_cash_balance reponly missing= -1 out=pos_cash_balance_cleaned;
var CNT_INSTALMENT CNT_INSTALMENT_FUTURE;
run;

proc export data=work.pos_cash_balance_cleaned
outfile='E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Data\pos_cash_balance_cleaned.csv'
	dbms=csv replace;
run;
PROC UNIVARIATE DATA=WORK.POS_CASH_BALANCE_CLEANED;
var	CNT_INSTALMENT CNT_INSTALMENT_FUTURE;
HISTOGRAM CNT_INSTALMENT CNT_INSTALMENT_FUTURE;
RUN;
/*INSTALMENT Missing*/
proc stdize data=work.installments_payments reponly missing= 0 out=installments_payments_cleaned;
var AMT_PAYMENT;
run;
proc stdize data=work.installments_payments_cleaned reponly missing= -9999 out=installments_payments_cleaned;
var DAYS_ENTRY_PAYMENT;
run;
proc export data=work.installments_payments_cleaned
outfile='E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Data\installments_payments_cleaned.csv'
	dbms=csv replace;
run;
/*Check the missing on credit card balance*/
PROC UNIVARIATE DATA=WORK.CREDIT_CARD_BALANCE;
var	AMT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT AMT_DRAWINGS_POS_CURRENT AMT_INST_MIN_REGULARITY AMT_PAYMENT_TOTAL_CURRENT AMT_PAYMENT_CURRENT CNT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_POS_CURRENT CNT_INSTALMENT_MATURE_CUM;
HISTOGRAM AMT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT AMT_DRAWINGS_POS_CURRENT AMT_INST_MIN_REGULARITY AMT_PAYMENT_TOTAL_CURRENT AMT_PAYMENT_CURRENT CNT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_POS_CURRENT CNT_INSTALMENT_MATURE_CUM;
RUN;

/*REPLACE ALL MISSING TO 0 in credit card balance*/
proc stdize data=work.credit_card_balance reponly missing= 0 out=credit_card_balance_cleaned;
var	AMT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT AMT_DRAWINGS_POS_CURRENT AMT_INST_MIN_REGULARITY AMT_PAYMENT_TOTAL_CURRENT AMT_PAYMENT_CURRENT CNT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_POS_CURRENT CNT_INSTALMENT_MATURE_CUM;
run;

proc export data=work.credit_card_balance_cleaned
outfile='E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Data\credit_card_balance_cleaned.csv'
	dbms=csv replace;
run;
PROC UNIVARIATE DATA=WORK.CREDIT_CARD_BALANCE_CLEANED;
var	AMT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT AMT_DRAWINGS_POS_CURRENT AMT_INST_MIN_REGULARITY AMT_PAYMENT_TOTAL_CURRENT AMT_PAYMENT_CURRENT CNT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_POS_CURRENT CNT_INSTALMENT_MATURE_CUM;
HISTOGRAM AMT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT AMT_DRAWINGS_POS_CURRENT AMT_INST_MIN_REGULARITY AMT_PAYMENT_TOTAL_CURRENT AMT_PAYMENT_CURRENT CNT_DRAWINGS_ATM_CURRENT AMT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_OTHER_CURRENT CNT_DRAWINGS_POS_CURRENT CNT_INSTALMENT_MATURE_CUM;
RUN;

/*Previous Application*/
data work.previous_application;
set work.previous_application;
if NAME_TYPE_SUITE in (' ', '.') then NAME_TYPE_SUITE = 'Other_C';
run;
data work.previous_application_cleaned;
set work.previous_application_cleaned;
if PRODUCT_COMBINATION in (' ', '.') then PRODUCT_COMBINATION = 'Unknown';
run;
data work.previous_application (DROP=DAYS_FIRST_DRAWING DAYS_FIRST_DUE DAYS_LAST_DUE DAYS_LAST_DUE_1ST_VERSION DAYS_TERMINATION NFLAG_INSURED_ON_APPROVAL RATE_DOWN_PAYMENT RATE_INTEREST_PRIMARY RATE_INTEREST_PRIMARY RATE_INTEREST_PRIVILEGED);
SET WORK.PREVIOUS_APPLICATION;
RUN;

proc stdize data=WORK.previous_application reponly missing= 0 out=previous_application_CLEANED;
var	AMT_ANNUITY AMT_CREDIT AMT_GOODS_PRICE AMT_DOWN_PAYMENT CNT_PAYMENT;
run;

proc export data=work.previous_application_CLEANED
outfile='E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Data\previous_application_cleaned.csv'
	dbms=csv replace;
run;

/*Application Train*/
data work.application_train;
set work.application_train;
if NAME_TYPE_SUITE in (' ', '.') then NAME_TYPE_SUITE = 'Other_C';
run;
data work.application_train;
set work.application_train;
array _a$ _character_;
do over _a;
	if missing(_a) then _a = 'Unknown';
end;
run;
proc stdize data=WORK.application_train reponly missing= 0 out=application_train;
var	AMT_ANNUITY AMT_GOODS_PRICE;
run;
PROC UNIVARIATE DATA=WORK.application_train;
var	AMT_REQ_CREDIT_BUREAU_HOUR AMT_REQ_CREDIT_BUREAU_DAY AMT_REQ_CREDIT_BUREAU_MON AMT_REQ_CREDIT_BUREAU_QRT AMT_REQ_CREDIT_BUREAU_WEEK AMT_REQ_CREDIT_BUREAU_YEAR;
histogram AMT_REQ_CREDIT_BUREAU_HOUR AMT_REQ_CREDIT_BUREAU_DAY AMT_REQ_CREDIT_BUREAU_MON AMT_REQ_CREDIT_BUREAU_QRT AMT_REQ_CREDIT_BUREAU_WEEK AMT_REQ_CREDIT_BUREAU_YEAR;
RUN;
proc stdize data=WORK.application_train reponly missing= 0 out=application_train;
var	AMT_REQ_CREDIT_BUREAU_HOUR AMT_REQ_CREDIT_BUREAU_DAY AMT_REQ_CREDIT_BUREAU_MON AMT_REQ_CREDIT_BUREAU_QRT AMT_REQ_CREDIT_BUREAU_WEEK AMT_REQ_CREDIT_BUREAU_YEAR;
run;
PROC UNIVARIATE DATA=WORK.application_train;
var	DEF_30_CNT_SOCIAL_CIRCLE DEF_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE OBS_60_CNT_SOCIAL_CIRCLE;
histogram DEF_30_CNT_SOCIAL_CIRCLE DEF_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE OBS_60_CNT_SOCIAL_CIRCLE;
RUN;
proc stdize data=WORK.application_train reponly missing= 0 out=application_train;
var	DEF_30_CNT_SOCIAL_CIRCLE DEF_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE OBS_60_CNT_SOCIAL_CIRCLE DAYS_LAST_PHONE_CHANGE;
run;

proc stdize data=WORK.application_train reponly missing= mean out=application_train_cleaned;
run;
proc export data=work.application_train_cleaned
outfile='E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Data\application_train_cleaned.csv'
	dbms=csv replace;
run;

/*Application Test*/

data work.application_test;
set work.application_test;
if NAME_TYPE_SUITE in (' ', '.') then NAME_TYPE_SUITE = 'Other_C';
run;
data work.application_test;
set work.application_test;
array _a$ _character_;
do over _a;
	if missing(_a) then _a = 'Unknown';
end;
run;
proc stdize data=WORK.application_test reponly missing= 0 out=application_test;
var	AMT_ANNUITY AMT_GOODS_PRICE;
run;
PROC UNIVARIATE DATA=WORK.application_test;
var	AMT_REQ_CREDIT_BUREAU_HOUR AMT_REQ_CREDIT_BUREAU_DAY AMT_REQ_CREDIT_BUREAU_MON AMT_REQ_CREDIT_BUREAU_QRT AMT_REQ_CREDIT_BUREAU_WEEK AMT_REQ_CREDIT_BUREAU_YEAR;
histogram AMT_REQ_CREDIT_BUREAU_HOUR AMT_REQ_CREDIT_BUREAU_DAY AMT_REQ_CREDIT_BUREAU_MON AMT_REQ_CREDIT_BUREAU_QRT AMT_REQ_CREDIT_BUREAU_WEEK AMT_REQ_CREDIT_BUREAU_YEAR;
RUN;
proc stdize data=WORK.application_test reponly missing= 0 out=application_test;
var	AMT_REQ_CREDIT_BUREAU_HOUR AMT_REQ_CREDIT_BUREAU_DAY AMT_REQ_CREDIT_BUREAU_MON AMT_REQ_CREDIT_BUREAU_QRT AMT_REQ_CREDIT_BUREAU_WEEK AMT_REQ_CREDIT_BUREAU_YEAR;
run;
PROC UNIVARIATE DATA=WORK.application_test;
var	DEF_30_CNT_SOCIAL_CIRCLE DEF_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE OBS_60_CNT_SOCIAL_CIRCLE;
histogram DEF_30_CNT_SOCIAL_CIRCLE DEF_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE OBS_60_CNT_SOCIAL_CIRCLE;
RUN;
proc stdize data=WORK.application_test reponly missing= 0 out=application_test;
var	DEF_30_CNT_SOCIAL_CIRCLE DEF_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE OBS_60_CNT_SOCIAL_CIRCLE DAYS_LAST_PHONE_CHANGE;
run;

proc stdize data=WORK.application_test reponly missing= mean out=application_test_cleaned;
run;
proc export data=work.application_test_cleaned
outfile='E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Data\application_test_cleaned.csv'
	dbms=csv replace;
run;