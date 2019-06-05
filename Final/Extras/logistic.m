feat_lecture = [13     1     1     1     1     1     1     1     1     1     1     3     1     3     3     1     1  1     3     2     5     6     1     1     1     1     2     3     9     1     4     1     4     8   4     3;35    33    40    39    30    30    59     4    27    45    64    38    45    52    44    27    32  44    59    94    50     4    63    62    40    54    25    48   102    56    67    57    74     4  55    67]';

labels_lecture = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]';

plot(feat_lecture(labels_lecture==0,1), feat_lecture(labels_lecture==0,2), 'r.', 'markersize', 20)
hold on
plot(feat_lecture(labels_lecture==1,1), feat_lecture(labels_lecture==1,2), 'g.', 'markersize', 20)
hold off

feat1 = feat_lecture(:,1);
feat2 = feat_lecture(:,2);


%%
% 2-fold CV Logistic regression

% This will look familiar, since I'm using the same general structure as I
% used for KNN & lDA

% First look at labels:  We want our data to be balanced within each fold
labels_lecture
% Logistic regression uses 1/0 as dependent variable, so we will change
%labels_lecture = labels_lecture -1;


% I'm being lazy here.  You should randomly select the data for your folds.
%  You will adapt the homework to do so.
%cv_groups = [ones(1, 25), 2*ones(1, 25), ones(1, 25), 2*ones(1, 25)];

g1= repmat([1:6], 1,3);
g1 = g1(randperm(18));
g2 = repmat([1:6], 1,3);
g2 = g2(randperm(18));
    
cv_groups = [g1, g2];



pred = zeros(size(labels_lecture)); % vector to hold predictions


for fold=1:6
  % generally, data should be normalized.  You will do this int he homework
  test = feat_lecture(cv_groups == fold,:);  
  train = feat_lecture(cv_groups ~= fold,:);
 
  labels_train = labels_lecture(cv_groups ~= fold);
  ntest = size(test, 1);
  ntrain = size(train, 1);
  pred_test = zeros(1, ntest);
  
  % Train the classifier (logistic model fit
  beta = glmfit(train, labels_train, 'binomial', 'link', 'logit');
  
  % Need to use the inverse logit to get the probabilities for test
  xb = [ones(size(test,1), 1), test]*beta;
  prob_test = exp(xb)./(1+exp(xb));
  pred_test = 1*prob_test>.5;
  
  pred(cv_groups == fold) = pred_test;
end

match = labels_lecture == pred;

accuracy_g1 = mean(match(labels_lecture == 0))
accuracy_g2 = mean(match(labels_lecture == 1))

